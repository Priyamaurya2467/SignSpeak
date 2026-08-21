import csv
from pathlib import Path

class LandmarkCollector:
    def __init__(self,output_dir="../dataset/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


    def save_landmarks(self,landmarks,label):
        if not landmarks:
            return

        label_dir = self.output_dir/label
        label_dir.mkdir(parents=True,exist_ok=True)

        file_path = label_dir/"landmarks.csv"
        file_exists = file_path.exists()

        with open(file_path,"a",newline="") as file:
            writer = csv.writer(file)

            if not file_exists:
                header=[]

                for i in range(21):
                    header.extend({
                        f"x{i}",
                        f"y{i}",
                        f"z{i}"
                    })

                    writer.writerow(header)

                row=[]

                for landmark in landmarks:
                    row.extend({
                        landmark.x,
                        landmark.y,
                        landmark.z
                    
                    })

                writer.writerow(row)