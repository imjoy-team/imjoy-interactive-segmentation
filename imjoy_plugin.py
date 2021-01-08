import os
import io
import time
import numpy as np
import json
from imageio import imread, imwrite
from data_utils import plot_images
from imjoy import api

from interactive_trainer import InteractiveTrainer
import asyncio


class ImJoyPlugin:
    def __init__(self, trainer):
        self._trainer = trainer

    async def setup(self):
        self.geojson_layer = None
        self.mask_layer = None
        self.image_layer = None
        self._mask_prediction = None

    def get_trainer(self):
        return self._trainer

    def start_training(self):
        self._trainer.start()

    def stop_training(self):
        self._trainer.stop()

    async def get_next_sample(self, sample_name=None, folder="test"):
        if self.image_layer:
            self.viewer.remove_layer(self.image_layer)
        if self.mask_layer:
            self.viewer.remove_layer(self.mask_layer)
        if self.geojson_layer:
            self.viewer.remove_layer(self.geojson_layer)
        self._mask_prediction = None
        if folder == "test":
            image, _, info = self._trainer.get_test_sample(sample_name)
        elif folder == "train":
            image, _, info = self._trainer.get_training_sample(sample_name)
        else:
            raise Exception("unsupported folder: " + folder)
        self.current_sample_info = info
        self.current_image = image
        self.image_layer = await self.viewer.view_image(
            (image * 255).astype("uint8"),
            type="itk-vtk",
            name=self.current_sample_info["name"],
        )

    async def get_augmentations(self):
        if self.mask_layer:
            self.viewer.remove_layer(self.mask_layer)
        if self.geojson_layer:
            self.viewer.remove_layer(self.geojson_layer)
        self._trainer.plot_augmentations_async()

        async def check_augmentations():
            self.viewer.set_loader(True)
            figure = self._trainer._plot_augmentations_result
            if figure is None:
                self.viewer.set_timeout(check_augmentations, 1500)
                return
            api.showMessage("augmentations done")
            try:
                self.image_layer = await self.viewer.view_image(
                    figure, type="itk-vtk", name="Augmented grid"
                )
            except Exception as e:
                api.showMessage(str(e))
            finally:
                self.viewer.set_loader(False)

        self.viewer.set_timeout(check_augmentations, 1000)

    async def predict(self):
        self.viewer.set_loader(True)
        if self.mask_layer:
            self.viewer.remove_layer(self.mask_layer)
        if self.geojson_layer:
            self.viewer.remove_layer(self.geojson_layer)
        self._trainer.predict_async(self.current_image)

        async def check_prediction():
            self.viewer.set_loader(True)
            result = self._trainer.get_prediction_result()
            if result is None:
                self.viewer.set_timeout(check_prediction, 500)
                return
            api.showMessage("prediction done")
            try:
                self.viewer.set_loader(True)
                polygons, mask = result
                mask = np.clip(mask * 255, 0, 255).astype("uint8")
                # imwrite(
                #     os.path.join(self.current_sample_info["path"], "prediction.png"),
                #     mask,
                # )
                self._mask_prediction = mask
                self.current_annotation = polygons

                self.mask_layer = await self.viewer.view_image(
                    mask,
                    type="itk-vtk",
                    name=self._trainer.object_name + "_mask",
                    opacity=0.5,
                )
                self.viewer.set_loader(False)
                if len(polygons) > 0:
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
            except Exception as e:
                api.showMessage(str(e))
            finally:
                self.viewer.set_loader(False)

        self.viewer.set_timeout(check_prediction, 1000)

    async def fetch_mask(self):
        if self.mask_layer:
            self.viewer.remove_layer(self.mask_layer)
        if self.geojson_layer:
            self.viewer.remove_layer(self.geojson_layer)
        with io.open(
            os.path.join(
                self._trainer.data_dir,
                "test",
                self.current_sample_info["name"],
                "annotation.json",
            ),
            "r",
            encoding="utf-8-sig",
        ) as myfile:
            polygons = json.load(myfile)
        size = polygons["bbox"][3]
        for i, feature in enumerate(polygons["features"]):
            coordinates = feature["geometry"]["coordinates"][0]
            new_coordinates = []
            for j, coordinate in enumerate(coordinates):
                x, y = coordinate
                if x < 0:
                    x = 0
                if x > size:
                    x = size
                if y < 0:
                    y = 0
                if y > size:
                    y = size
                y = size - y
                new_coordinates.append([x, y])
            polygons["features"][i]["geometry"]["coordinates"][0] = new_coordinates

        polygons = list(
            map(
                lambda feature: np.array(
                    feature["geometry"]["coordinates"][0], dtype=np.uint16
                ),
                polygons["features"],
            )
        )
        mask_file_name = (
            self._trainer.object_name + "_" + self._trainer.mask_type + ".png"
        )
        mask = imread(
            os.path.join(
                self._trainer.data_dir,
                "test",
                self.current_sample_info["name"],
                mask_file_name,
            )
        )

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
        if self._mask_prediction is None:
            api.showMessage("please predict first")
            return
        if not self.geojson_layer:
            api.showMessage("no annotation available")
            return

        self.current_annotation = await self.geojson_layer.get_features()
        if len(self.current_annotation["features"]) < 1:
            api.showMessage("no annotation available")
            return

        img = imread(
            os.path.join(
                self._trainer.data_dir,
                "test",
                self.current_sample_info["name"],
                self._trainer.input_channels[0],
            )
        )
        size = img.shape[1]
        for i, feature in enumerate(self.current_annotation["features"]):
            coordinates = feature["geometry"]["coordinates"][0]
            new_coordinates = []
            for j, coordinate in enumerate(coordinates):
                x, y = coordinate
                if x < 0:
                    x = 0
                if x > size:
                    x = size
                if y < 0:
                    y = 0
                if y > size:
                    y = size
                y = size - y
                new_coordinates.append([x, y])
            self.current_annotation["features"][i]["geometry"]["coordinates"][
                0
            ] = new_coordinates
        self._trainer.push_sample_async(
            self.current_sample_info["name"],
            self.current_annotation,
            target_folder="train",
            prediction=self._mask_prediction,
        )
        self._mask_prediction = None
        # api.showMessage("Sample moved to the training set")
        if self.geojson_layer:
            self.viewer.remove_layer(self.geojson_layer)
        if self.mask_layer:
            self.viewer.remove_layer(self.mask_layer)

    async def send_for_evaluation(self):
        self.current_annotation = await self.geojson_layer.get_features()
        self._trainer.push_sample_async(
            self.current_sample_info["name"],
            self.current_annotation,
            target_folder="valid",
            prediction=self._mask_prediction,
        )
        if self.geojson_layer:
            self.viewer.remove_layer(self.geojson_layer)
        if self.mask_layer:
            self.viewer.remove_layer(self.mask_layer)

    def get_sample_list(self, group):
        data_dir = os.path.join(self._trainer.data_dir, group)
        samples = []
        for sf in os.listdir(data_dir):
            sfd = os.path.join(data_dir, sf)
            if os.path.isdir(sfd) and not sf.startswith("."):
                samples.append({"title": sf, "isLeaf": True, "data": {"group": group}})
        return samples

    async def run(self, ctx):
        self.viewer = await api.createWindow(
            src="https://kaibu.org/#/app", fullscreen=True
        )
        self.viewer.set_loader(True)

        async def node_dbclick_callback(node):
            await self.get_next_sample(node["title"], node["data"]["group"])

        tree = await self.viewer.add_widget(
            {
                "_rintf": True,
                "type": "tree",
                "name": "Samples",
                "node_dbclick_callback": node_dbclick_callback,
                "nodes": [
                    {
                        "title": "test",
                        "isLeaf": False,
                        "children": self.get_sample_list("test"),
                        "isExpanded": False,
                    },
                    {
                        "title": "train",
                        "isLeaf": False,
                        "children": self.get_sample_list("train"),
                        "isExpanded": False,
                    },
                ],
            }
        )

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
                    {
                        "type": "button",
                        "label": "Predict",
                        "callback": self.predict,
                    },
                    {
                        "type": "button",
                        "label": "Fetch the Mask",
                        "callback": self.fetch_mask,
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
                    {
                        "type": "button",
                        "label": "Get Augmented Patches",
                        "callback": self.get_augmentations,
                    },
                ],
            }
        )

        losses = self._trainer.reports or []
        if len(losses) > 10000:
            step = int(len(losses) / 5000)
            losses = losses[::step]
        chart = await self.viewer.add_widget(
            {
                "_rintf": True,
                "name": "Training",
                "type": "vega",
                "schema": {
                    "$schema": "https://vega.github.io/schema/vega-lite/v4.8.1.json",
                    "encoding": {
                        "x": {"type": "quantitative", "field": "iteration"},
                        "y": {"type": "quantitative", "field": "loss"},
                        "tooltip": [
                            {"field": "loss", "type": "quantitative"},
                            {"field": "rolling_mean", "type": "quantitative"},
                            {"field": "iteration", "type": "ordinal"},
                        ],
                    },
                    "transform": [
                        {
                            "window": [
                                {"field": "loss", "op": "mean", "as": "rolling_mean"}
                            ],
                            "frame": [-200, 200],
                        }
                    ],
                    "layer": [
                        {
                            "mark": {"type": "line", "opacity": 0.3},
                            "encoding": {"y": {"field": "loss", "title": "Loss"}},
                        },
                        {
                            "mark": {"type": "line", "color": "red", "size": 3},
                            "encoding": {"y": {"field": "rolling_mean"}},
                        },
                    ],
                    "data": {"name": "loss"},
                    "datasets": {"loss": losses},
                },
            }
        )

        self.last_iteration = None

        async def refresh():
            if len(self._trainer.reports) > 0:
                v = self._trainer.reports[-1]
                error = self._trainer.get_error()
                if v["iteration"] != self.last_iteration:
                    title = f'Iteration {v["iteration"]}, Loss: {v["loss"]:.4f}'
                    if not self._trainer.training_enabled:
                        title += " (stopped)"
                    else:
                        if error:
                            title += str(error)
                    await chart.set_title(title)
                    await chart.append("loss", v)
                    self.last_iteration = v["iteration"]
                elif self._trainer.training_enabled:
                    if error:
                        await chart.set_title(f"Error: {error}")
                    else:
                        await chart.set_title("Starting...")
                else:
                    if error:
                        await chart.set_title(f"Error: {error}")
                        await api.error(f"Error: {error}")
                        # await api.showMessage(f"Error: {error}")
            self.viewer.set_timeout(refresh, 2000)

        self.viewer.set_timeout(refresh, 2000)
        self.viewer.set_loader(False)


def start_interactive_segmentation(*args, **kwargs):
    trainer = InteractiveTrainer.get_instance(*args, **kwargs)
    plugin = ImJoyPlugin(trainer)
    api.export(plugin)
    return plugin
