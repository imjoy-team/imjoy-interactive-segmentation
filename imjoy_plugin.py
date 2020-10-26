import os
import time
from imjoy import api
from interactive_trainer import InteractiveTrainer

trainer = InteractiveTrainer(
    "./data/hpa_dataset_v2",
    ["microtubules.png", "er.png", "nuclei.png"],
    object_name="cell",
)


class ImJoyPlugin:
    async def setup(self):
        pass

    def start_training(self):
        trainer.start()

    async def get_next_sample(self):
        image, _, info = trainer.get_test_sample()
        api.alert(str(image.shape))
        self.current_sample_name = info["name"]
        self.current_image = image
        await self.viewer.view_image(
            image, type="itk-vtk", name=self.current_sample_name
        )

    async def predict(self):
        polygons = trainer.predict(self.current_image)
        self.current_annotation = polygons
        self.geojson_layer = await self.viewer.add_shapes(
            polygons, shape_type="polygon", edge_color="red", name=trainer.object_name
        )

    async def send_for_training(self):
        self.current_annotation = await self.geojson_layer.get_features()
        trainer.push_sample(
            self.current_sample_name, self.current_annotation, target_folder="train"
        )

    async def send_for_evaluation(self):
        self.current_annotation = await self.geojson_layer.get_features()
        trainer.push_sample(
            self.current_sample_name, self.current_annotation, target_folder="valid"
        )

    async def run(self, ctx):
        self.viewer = await api.createWindow(src="https://kaibu.org/#/app")
        await self.viewer.set_ui(
            {
                "_rintf": True,
                "title": "Utilities",
                "elements": [
                    {
                        "type": "button",
                        "label": "Next Image",
                        "callback": self.get_next_sample,
                    },
                    {
                        "type": "button",
                        "label": "Start Training",
                        "callback": self.start_training,
                    },
                    {
                        "type": "button",
                        "label": "Predict",
                        "callback": self.predict,
                    },
                    {
                        "type": "button",
                        "label": "Send for Training",
                        "callback": self.send_for_training,
                    },
                    {
                        "type": "button",
                        "label": "Send for Evaluation",
                        "callback": self.send_for_evaluation,
                    },
                ],
            }
        )


def run_interactive_ml():
    api.export(ImJoyPlugin())
