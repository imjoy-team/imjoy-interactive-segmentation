import os
import time
from imjoy import api
from interactive_trainer import InteractiveTrainer


class ImJoyPlugin:
    def __init__(self, trainer):
        self._trainer = trainer

    async def setup(self):
        self.geojson_layer = None
        self.mask_layer = None
        self.image_layer = None

    def start_training(self):
        self._trainer.start()

    def stop_training(self):
        self._trainer.stop()

    async def get_next_sample(self):
        if self.image_layer:
            self.viewer.remove_layer(self.image_layer)
        image, _, info = self._trainer.get_test_sample()
        self.current_sample_name = info["name"]
        self.current_image = image
        self.image_layer = await self.viewer.view_image(
            (image * 255).astype("uint8"), type="itk-vtk", name=self.current_sample_name
        )

    async def predict(self):
        if self.mask_layer:
            self.viewer.remove_layer(self.mask_layer)
        if self.geojson_layer:
            self.viewer.remove_layer(self.geojson_layer)
        polygons, mask = self._trainer.predict(self.current_image)
        self.current_annotation = polygons
        if len(polygons) > 0:
            self.mask_layer = await self.viewer.view_image(
                mask,
                type="itk-vtk",
                name=self._trainer.object_name + "_mask",
                opacity=0.5,
            )
            if len(polygons) < 2000:
                self.geojson_layer = await self.viewer.add_shapes(
                    polygons,
                    shape_type="polygon",
                    edge_color="red",
                    name=self._trainer.object_name,
                )
            else:
                api.showMessage(f"Too many object detected ({len(polygons)}).")
        else:
            api.showMessage("No object detected.")

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

    async def run(self, ctx):
        self.viewer = await api.createWindow(src="https://kaibu.org/#/app")
        self.viewer.set_loader(True)
        await self.viewer.add_widget(
            {
                "_rintf": True,
                "name": "Control",
                "type": "control",
                "elements": [
                    {
                        "type": "button",
                        "label": "Start Training",
                        "callback": self.start_training,
                    },
                    {
                        "type": "button",
                        "label": "Stop Training",
                        "callback": self.stop_training,
                    },
                    {
                        "type": "button",
                        "label": "Get an Image",
                        "callback": self.get_next_sample,
                    },
                    {"type": "button", "label": "Predict", "callback": self.predict,},
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
        losses = self._trainer.reports or []
        chart = await self.viewer.add_widget(
            {
                "_rintf": True,
                "name": "Training",
                "type": "vega",
                "schema": {
                    "$schema": "https://vega.github.io/schema/vega-lite/v4.8.1.json",
                    "mark": "line",
                    "encoding": {
                        "x": {"type": "quantitative", "field": "iteration"},
                        "y": {"type": "quantitative", "field": "loss"},
                    },
                    "data": {"name": "loss"},
                    "datasets": {"loss": losses},
                },
            }
        )

        self.last_iteration = None

        async def refresh():
            if len(self._trainer.reports) > 0:
                v = self._trainer.reports[-1]
                if v["iteration"] != self.last_iteration:
                    title = f'Iteration {v["iteration"]}, Loss: {v["loss"]:.4f}'
                    if not self._trainer.training_enabled:
                        title += " (stopped)"
                    await chart.set_title(title)
                    await chart.append("loss", v)
                    self.last_iteration = v["iteration"]
                elif self._trainer.training_enabled:
                    await chart.set_title("Starting...")
            self.viewer.set_timeout(refresh, 2000)

        self.viewer.set_timeout(refresh, 2000)
        self.viewer.set_loader(False)


def start_interactive_segmentation(*args, **kwargs):
    trainer = InteractiveTrainer.get_instance(*args, **kwargs)
    api.export(ImJoyPlugin(trainer))
