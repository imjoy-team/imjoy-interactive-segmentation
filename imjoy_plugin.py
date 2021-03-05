import os
import io
import time
import numpy as np
import json
import asyncio
import traceback
from imageio import imread, imwrite
from data_utils import plot_images
from imjoy import api


from interactive_trainer import InteractiveTrainer, byte_scale

from imgseg.geojson_utils import geojson_to_masks


class ImJoyPlugin:
    def __init__(self, trainer, restore_test_annotation=False, auto_get_sample=True):
        self._trainer = trainer
        self.restore_test_annotation = restore_test_annotation
        self.auto_get_sample = auto_get_sample

    async def setup(self):
        self.geojson_layer = None
        self.mask_layer = None
        self.image_layer = None
        self._mask_prediction = None
        self.tree = None

    def get_trainer(self):
        return self._trainer

    def start_training(self):
        self._trainer.start()

    def stop_training(self):
        self._trainer.stop()

    async def get_next_sample(self, sample_name=None, folder="test"):
        try:
            if self.image_layer:
                self.viewer.remove_layer(self.image_layer)
            if self.mask_layer:
                self.viewer.remove_layer(self.mask_layer)
            if self.geojson_layer:
                self.viewer.remove_layer(self.geojson_layer)
            self._mask_prediction = None
            if folder == "test":
                image, geojson_annotation, info = self._trainer.get_test_sample(sample_name)
            elif folder == "train":
                image, geojson_annotation, info = self._trainer.get_training_sample(
                    sample_name
                )
            else:
                raise Exception("unsupported folder: " + folder)
            self.current_sample_info = info
            self.current_image = image
            self.image_layer = await self.viewer.view_image(
                (byte_scale(image)).astype("uint8"),
                type="itk-vtk",
                name=self.current_sample_info["name"],
            )
            self.geojson_layer = await self.viewer.add_shapes(
                [], shape_type="polygon", edge_color="red", name=self._trainer.object_name,
            )
            # don't restore annotation for test if restore_test_annotation=False
            if folder == "test" and not self.restore_test_annotation:
                return
            if geojson_annotation is not None:
                size = image.shape[1]
                geojson_annotation = self.flipud_annotation(geojson_annotation, size)
                await self.geojson_layer.set_features(geojson_annotation)
        except Exception as e:
            await api.error(traceback.format_exc())
            await api.alert("Failed to load sample, error:" + str(e))
        finally:
            self.viewer.set_loader(False)

    async def test_augmentations(self):
        if self.image_layer:
            self.viewer.remove_layer(self.image_layer)
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
            api.showMessage("Testing augmentation done")
            try:
                self.image_layer = await self.viewer.view_image(
                    figure, type="itk-vtk", name="Augmented images"
                )
            except Exception as e:
                api.showMessage(str(e))
            finally:
                self.viewer.set_loader(False)

        self.viewer.set_timeout(check_augmentations, 1000)

    async def predict(self):
        if not self.image_layer:
            api.showMessage("No selected image for prediction")
            return
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
                mask = ((mask > 0) * 255).astype("uint8")
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
                    visible=False,
                )
                self.viewer.set_loader(False)
                if len(polygons) > 0:
                    if len(polygons) > 2000:
                        polygons = polygons[:2000]
                        api.showMessage(
                            f"WARNING: Too many object detected ({len(polygons)}), only displaying the first 2000 objects."
                        )
                else:
                    polygons = []
                    api.showMessage("No object detected.")
                self.geojson_layer = await self.viewer.add_shapes(
                    polygons,
                    shape_type="polygon",
                    edge_color="red",
                    name=self._trainer.object_name,
                )
            except Exception as e:
                api.error(traceback.format_exc())
                api.showMessage(str(e))
            finally:
                self.viewer.set_loader(False)

        self.viewer.set_timeout(check_prediction, 1000)

    async def fetch_mask(self):
        if self.mask_layer:
            self.viewer.remove_layer(self.mask_layer)
        if self.geojson_layer:
            self.viewer.remove_layer(self.geojson_layer)
        annotation_file = os.path.join(
            self._trainer.data_dir,
            self.current_sample_info["folder"],
            self.current_sample_info["name"],
            "annotation.json",
        )
        with io.open(annotation_file, "r", encoding="utf-8-sig",) as myfile:
            polygons = json.load(myfile)

        polygons = self.flipud_annotation(polygons)
        polygons = list(
            map(
                lambda feature: np.array(
                    feature["geometry"]["coordinates"][0], dtype=np.uint16
                ),
                polygons["features"],
            )
        )
        mask_dict = geojson_to_masks(annotation_file, mask_types=["labels"])
        labels = mask_dict["labels"]
        mask = ((labels > 0) * 255).astype("uint8")

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
        if not self.geojson_layer:
            api.showMessage("no annotation available")
            return

        await self.save_annotation()
        self._trainer.push_sample_async(
            self.current_sample_info["name"],
            self.current_annotation,
            target_folder="train",
            prediction=self._mask_prediction,
        )
        self._mask_prediction = None
        
        if self.image_layer:
            self.viewer.remove_layer(self.image_layer)
        if self.geojson_layer:
            self.viewer.remove_layer(self.geojson_layer)
        if self.mask_layer:
            self.viewer.remove_layer(self.mask_layer)
        await self.update_file_tree()
        await api.showMessage("Sample moved to the training set")
        if self.auto_get_sample:
            await self.get_next_sample()

    async def send_for_evaluation(self):
        self.current_annotation = await self.geojson_layer.get_features()
        self._trainer.push_sample_async(
            self.current_sample_info["name"],
            self.current_annotation,
            target_folder="valid",
            prediction=self._mask_prediction,
        )
        if self.image_layer:
            self.viewer.remove_layer(self.image_layer)
        if self.geojson_layer:
            self.viewer.remove_layer(self.geojson_layer)
        if self.mask_layer:
            self.viewer.remove_layer(self.mask_layer)
        await self.update_file_tree()
        if self.auto_get_sample:
            await self.get_next_sample()

    def get_sample_list(self, group):
        data_dir = os.path.join(self._trainer.data_dir, group)
        samples = []
        for sf in os.listdir(data_dir):
            sfd = os.path.join(data_dir, sf)
            if os.path.isdir(sfd) and not sf.startswith("."):
                samples.append({"title": sf, "isLeaf": True, "data": {"group": group}})
        return samples

    async def update_file_tree(self):
        await self.tree.clear_nodes()
        nodes = [
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
        ]
        await self.tree.set_nodes(nodes)

    async def reload_samples(self):
        try:
            self._trainer.reload_sample_pool()
            await self.update_file_tree()
            await api.showMessage("Sample pool reloaded.")
        except Exception as e:
            await api.showMessage("Failed to reload samples, error: " + str(e))

    def flipud_annotation(self, annotation, size=None):
        size = size or annotation["bbox"][3]
        for i, feature in enumerate(annotation["features"]):
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
            annotation["features"][i]["geometry"]["coordinates"][0] = new_coordinates
        return annotation

    async def save_annotation(self):
        self.current_annotation = await self.geojson_layer.get_features()
        if len(self.current_annotation["features"]) < 1:
            api.showMessage("no annotation available")
            return

        img = imread(
            os.path.join(
                self._trainer.data_dir,
                self.current_sample_info["folder"],
                self.current_sample_info["name"],
                self._trainer.input_channels[0],
            )
        )
        size = img.shape[1]
        self.current_annotation = self.flipud_annotation(self.current_annotation, size)
        self._trainer.save_annotation_async(
            self.current_sample_info["folder"],
            self.current_sample_info["name"],
            self.current_annotation,
        )
        if self.current_sample_info["folder"] == 'train':
            self.reload_samples()
            await api.showMessage("Training sample pool reloaded.")

    async def run(self, ctx):
        self.viewer = await api.createWindow(
            src="https://kaibu.org/#/app", fullscreen=True
        )
        self.viewer.set_loader(True)
        await self.viewer.add_widget(
            {
                "_rintf": True,
                "name": "Control",
                "type": "control",
                "elements": [
                    {
                        "type": "button",
                        "label": "Get an Image",
                        "callback": self.get_next_sample,
                    },
                    {"type": "button", "label": "Predict", "callback": self.predict,},
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
                        "label": "Save Annotation",
                        "callback": self.save_annotation,
                    },
                    # {
                    #     "type": "button",
                    #     "label": "Reload Samples",
                    #     "callback": self.reload_samples,
                    # },
                    # {
                    #     "type": "button",
                    #     "label": "Fetch the Mask",
                    #     "callback": self.fetch_mask,
                    # },
                    {
                        "type": "button",
                        "label": "Send for Training",
                        "callback": self.send_for_training,
                    },
                    # {
                    #     "type": "button",
                    #     "label": "Send for Evaluation",
                    #     "callback": self.send_for_evaluation,
                    # },
                    {
                        "type": "button",
                        "label": "Test Augmentation",
                        "callback": self.test_augmentations,
                    },
                ],
            }
        )

        async def node_dbclick_callback(node):
            await self.get_next_sample(node["title"], node["data"]["group"])

        self.tree = await self.viewer.add_widget(
            {
                "_rintf": True,
                "type": "tree",
                "name": "Samples",
                "node_dbclick_callback": node_dbclick_callback,
                "nodes": [],
            }
        )
        await self.update_file_tree()

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
            error = self._trainer.get_error(clear=True)
            if len(self._trainer.reports) > 0:
                v = self._trainer.reports[-1]

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

            if error:
                await chart.set_title(f"Error: {error}")
                await api.error(f"Error: {error}")
                await api.showMessage(f"Error: {error}")
            self.viewer.set_timeout(refresh, 2000)

        self.viewer.set_timeout(refresh, 2000)
        self.viewer.set_loader(False)


def start_interactive_segmentation(*args, restore_test_annotation=False, auto_get_sample=True, **kwargs):
    trainer = InteractiveTrainer.get_instance(*args, **kwargs)
    plugin = ImJoyPlugin(trainer, restore_test_annotation, auto_get_sample)
    api.export(plugin)
    return plugin
