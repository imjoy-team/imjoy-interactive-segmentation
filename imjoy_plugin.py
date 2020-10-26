import os
import time
from imjoy import api
from interactive_trainer import InteractiveTrainer


class ImJoyPlugin:
    def __init__(self, trainer):
        self._trainer = trainer

    async def setup(self):
        pass

    def start_training(self):
        self._trainer.start()

    def stop_training(self):
        self._trainer.stop()

    async def get_next_sample(self):
        image, _, info = self._trainer.get_test_sample()
        self.current_sample_name = info["name"]
        self.current_image = image
        await self.viewer.view_image(
            (image * 255).astype("uint8"), type="itk-vtk", name=self.current_sample_name
        )

    async def predict(self):
        polygons = self._trainer.predict(self.current_image)
        self.current_annotation = polygons
        self.geojson_layer = await self.viewer.add_shapes(
            polygons,
            shape_type="polygon",
            edge_color="red",
            name=self._trainer.object_name,
        )

    async def send_for_training(self):
        self.current_annotation = await self.geojson_layer.get_features()
        self._trainer.push_sample(
            self.current_sample_name, self.current_annotation, target_folder="train"
        )

    async def send_for_evaluation(self):
        self.current_annotation = await self.geojson_layer.get_features()
        self._trainer.push_sample(
            self.current_sample_name, self.current_annotation, target_folder="valid"
        )
    
    def show_progress(self):
        if len(self._trainer.reports) > 0:
            api.showMessage(str(self._trainer.reports[-1]))
        else:
            api.showMessage('No progress yet.')

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
                        "label": "Show Progress",
                        "callback": self.show_progress,
                    },
                    {
                        "type": "button",
                        "label": "Stop Training",
                        "callback": self.stop_training,
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


def start_interactive_segmentation(*args, **kwargs):
    trainer = InteractiveTrainer.get_instance(*args, **kwargs)
    api.export(ImJoyPlugin(trainer))
