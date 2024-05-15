import re
import numpy as np
import cv2
import csv
import yaml
import json
from pathlib import Path
from natsort import natsorted
from loguru import logger
from tqdm import tqdm
from fire import Fire

# some scenes are skipped because they only contain ignored instances
SKIP_LIST = ["d7d40d4e-7a5d-2b36-97e7-34324c52ac42", "6bde6043-9162-246f-8e11-613aba0df55c"]

class RioPreprocessing:
    def __init__(
        self,
        data_dir: str = "./data/raw/rio",
        save_dir: str = "./data/processed/rio",
        modes: tuple = ("train", "validation", "test"),
        git_repo_dir: str = "./data/raw/3RScan",
        scene_metadata_path: str = "./data/raw/3RScan.json",
    ):
        self.data_dir = Path(data_dir)
        self.save_dir = Path(save_dir)
        self.modes = modes
        self.git_repo_dir = Path(git_repo_dir)
        self.scene_metadata_path = Path(scene_metadata_path)

        if not self.data_dir.exists():
            logger.error("Data folder doesn't exist")
            raise FileNotFoundError
        if self.save_dir.exists() is False:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        if not self.git_repo_dir.exists():
            logger.error("3RScan git repo doesn't exist")
            raise FileNotFoundError
        if not self.scene_metadata_path.exists():
            logger.error("3RScan.json scene metadata doesn't exist")
            raise FileNotFoundError

        self.files = {}
        for data_type in self.modes:
            self.files.update({data_type: []})

        self.config = self._load_yaml("conf/rio.yaml")
        self.create_label_database("conf/rio.yaml")
        self.pose = {mode: [] for mode in modes}

        for mode in modes:
            mode_dir = self.save_dir / mode
            if not mode_dir.exists():
                mode_dir.mkdir(parents=True, exist_ok=True)

        scene_metadata = self._load_json(scene_metadata_path)
        for i, scene in enumerate(scene_metadata):
            mode = scene["type"]
            if mode not in modes or scene["reference"] in SKIP_LIST:
                continue
            reference_filepath = scene["reference"]
            for j, scan in enumerate(scene["scans"]):
                filepaths = [
                    str(self.data_dir / reference_filepath),
                    str(self.data_dir / scan["reference"])
                ]
                self.files[mode].extend(filepaths)
                self.pose[mode].append([
                    parse_pose(scene),
                    parse_pose(scan)
                ])
        
        self.rio_to_scannet_label = {}
        with open(self.git_repo_dir / "data" / "mapping.tsv") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader) # skip header
            for row in reader:
                self.rio_to_scannet_label[row[1]] = int(row[2])

    def preprocess(self):
        for mode in self.modes:
            database = []
            for i, filepath in enumerate(tqdm(self.files[mode], unit="file")):
                filebase = self.process_file(filepath, i, mode)
                database.append(filebase)
            self.save_database(database, mode)
        self.joint_database()

    def make_instance_database(self):
        train_database = self._load_yaml(self.save_dir / "train_database.yaml")
        instance_database = {}
        for sample in tqdm(train_database):
            instances = self.extract_instance_from_file(sample)
            for instance in instances:
                scene = instance["scene"]
                panoptic_label = instance["panoptic_label"]
                unique_identifier = f"{scene}_{panoptic_label}"
                if unique_identifier in instance_database:
                    instance_database[unique_identifier]["filepaths"].append(instance["instance_filepath"])
                else:
                    instance_database[unique_identifier] = {
                        "semantic_label": instance["semantic_label"],
                        "filepaths": [instance["instance_filepath"]],
                    }
        self.save_database(list(instance_database.values()), "train_instances")

        validation_database = self._load_yaml(self.save_dir / "validation_database.yaml")
        for sample in tqdm(validation_database):
            instances = self.extract_instance_from_file(sample)
            for instance in instances:
                scene = instance["scene"]
                panoptic_label = instance["panoptic_label"]
                unique_identifier = f"{scene}_{panoptic_label}"
                if unique_identifier in instance_database:
                    instance_database[unique_identifier]["filepaths"].append(instance["instance_filepath"])
                else:
                    instance_database[unique_identifier] = {
                        "semantic_label": instance["semantic_label"],
                        "filepaths": [instance["instance_filepath"]],
                    }
        self.save_database(list(instance_database.values()), "trainval_instances")

    def extract_instance_from_file(self, sample):
        points = np.fromfile(sample["filepath"], dtype=np.float32).reshape(-1, 4)
        pose = np.array(sample["pose"]).T
        points[:, :3] = points[:, :3] @ pose[:3, :3] + pose[3, :3]
        label = np.fromfile(sample["label_filepath"], dtype=np.uint32)
        scene, sub_scene = re.search(r"(\d{2}).*(\d{6})", sample["filepath"]).group(1, 2)
        file_instances = []
        for panoptic_label in np.unique(label):
            semantic_label = panoptic_label & 0xFFFF
            semantic_label = np.vectorize(self.config["learning_map"].__getitem__)(semantic_label)
            if np.isin(semantic_label, range(1, 9)):
                instance_mask = label == panoptic_label
                instance_points = points[instance_mask, :]
                filename = f"{scene}_{panoptic_label:010d}_{sub_scene}.npy"
                instance_filepath = self.save_dir / "instances" / filename
                instance = {
                    "scene": scene,
                    "sub_scene": sub_scene,
                    "panoptic_label": f"{panoptic_label:010d}",
                    "instance_filepath": str(instance_filepath),
                    "semantic_label": semantic_label.item(),
                }
                if not instance_filepath.parent.exists():
                    instance_filepath.parent.mkdir(parents=True, exist_ok=True)
                np.save(instance_filepath, instance_points.astype(np.float32))
                file_instances.append(instance)
        return file_instances

    def save_database(self, database, mode):
        for element in database:
            self._dict_to_yaml(element)
        self._save_yaml(self.save_dir / (mode + "_database.yaml"), database)

    def joint_database(self, train_modes=["train", "validation"]):
        joint_db = []
        for mode in train_modes:
            joint_db.extend(self._load_yaml(self.save_dir / (mode + "_database.yaml")))
        self._save_yaml(self.save_dir / "trainval_database.yaml", joint_db)

    @classmethod
    def _save_yaml(cls, path, file):
        with open(path, "w") as f:
            yaml.safe_dump(file, f, default_style=None, default_flow_style=False)

    @classmethod
    def _dict_to_yaml(cls, dictionary):
        if not isinstance(dictionary, dict):
            return
        for k, v in dictionary.items():
            if isinstance(v, dict):
                cls._dict_to_yaml(v)
            if isinstance(v, np.ndarray):
                dictionary[k] = v.tolist()
            if isinstance(v, Path):
                dictionary[k] = str(v)

    @classmethod
    def _load_yaml(cls, filepath):
        with open(filepath) as f:
            file = yaml.safe_load(f)
        return file
    
    @classmethod
    def _load_json(self, filepath):
        with open(filepath) as f:
            file = json.load(f)
        return file

    def create_label_database(self, config_file):
        if (self.save_dir / "label_database.yaml").exists():
            return self._load_yaml(self.save_dir / "label_database.yaml")
        config = self._load_yaml(config_file)
        label_database = {}
        for key, old_key in config["learning_map_inv"].items():
            label_database.update(
                {
                    key: {
                        "name": config["labels"][old_key],
                        "color": config["color_map"][old_key][::-1],
                        "validation": not config["learning_ignore"][key],
                    }
                }
            )

        self._save_yaml(self.save_dir / "label_database.yaml", label_database)
        return label_database

    def process_file(self, filepath, index, mode):
        filepath = Path(filepath)
        scene = index // 2
        sub_scene = index % 2
        processed_filepath = self.save_dir / mode / f"{filepath.name}.npy"
        if processed_filepath.exists():
            sample = {
                "filepath": str(processed_filepath),
                "scene": scene,
                "pose": self.pose[mode][scene][sub_scene].tolist(),
            }
            return sample

        coords, features = load_obj_with_normals(filepath)
        labels = np.full((coords.shape[0], 2), 0)

        if mode in ["train", "validation"]:
            # 3RScan labels are given per segment: points -> segments -> labels
            instance_info_filepath = filepath / "semseg.v2.json"
            segment_indexes_filepath = next(filepath.glob("*.segs.v2.json"))
            instances = self._load_json(instance_info_filepath)
            segments = self._load_json(segment_indexes_filepath)
            segments = np.array(segments["segIndices"])

            for instance in instances["segGroups"]:
                occupied_segments = np.array(instance["segments"])
                occupied_points = np.isin(segments, occupied_segments)
                
                scannet_label = self.rio_to_scannet_label.get(instance["label"], -1)
                labels[occupied_points, 0] = scannet_label
                labels[occupied_points, 1] = instance["id"]

        points = np.hstack([coords, features, labels])

        processed_filepath = self.save_dir / mode / f"{filepath.name}.npy"
        np.save(processed_filepath, points.astype(np.float32))
        
        sample = {
            "filepath": str(processed_filepath),
            "scene": scene,
            "pose": self.pose[mode][scene][sub_scene].tolist(),
        }

        return sample


def parse_pose(scan):
    if "transform" in scan:
        pose = np.array(scan["transform"]).reshape(4, 4).T
    else:
        # transforms are not provided for test scenes
        pose = np.eye(4)
    return pose

def load_obj_with_normals(filepath):
    # the rio files have some missing edges, resulting in missing points when using open3d or trimesh
    # since we only care about the coordinates, normals and colors we can extract these values directly
    texture = cv2.imread(str(filepath / "mesh.refined_0.png"))
    # OpenCV uses BGR as its default colour order for images, but we want RGB
    # see https://stackoverflow.com/questions/39316447/opencv-giving-wrong-color-to-colored-images-on-loading
    texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
    height, width, _ = texture.shape
    coords, normals, colors = [], [], []
    mesh_filepath = filepath / "mesh.refined.v2.obj"
    with open(str(mesh_filepath)) as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith("v "):
                # the line defines the coordinates of a point
                # v 0. 1. 0.
                coords.append([float(x) for x in line.split(" ")[1:]])
            elif line.startswith("vn"):
                # the line defines the normals of a point
                # vn 0. 1. 0.
                normals.append([float(x) for x in line.split(" ")[1:]])
            elif line.startswith("vt"):
                # the line defines the color of a point
                # vt 0.5 0.5
                # u, v describe the location of the color in the texture, normalized to [0, 1]
                # u is the horizontal, v the vertical location
                u, v = line.split(" ")[1:]
                u, v = float(u), float(v)
                # sometimes the texture coordinates are not in [0, 1]
                # in this case we just choose the color black (255, 255, 255)
                if u < 0 or u > 1 or v < 0 or v > 1:
                    colors.append([255, 255, 255])
                    continue
                # scale with width/height to determine indices
                x = round(u*(width-1))
                y = round((1-v)*(height-1))
                colors.append(texture[y, x])

    coords = np.array(coords)
    normals = np.array(normals)
    colors = np.array(colors)
    feats = np.hstack((colors, normals))

    return coords, feats

def generate_sequence_name(scene_id, rescan_id):
    # e.g. (11, 2) -> "0011_02"
    return f'{str(scene_id).zfill(4)}_{str(rescan_id).zfill(2)}'

if __name__ == "__main__":
    Fire(RioPreprocessing)
