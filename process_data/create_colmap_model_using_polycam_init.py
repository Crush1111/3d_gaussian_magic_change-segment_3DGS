import json 

class polycam_for_colmap_init:
    
    def __init__(self, transform_json) -> None:
        self.pose_path = transform_json
    
    def load_pose_json(self):
        with open(self.pose_path, 'r') as f:
            meta = json.load(f)
