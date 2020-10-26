

class ImJoyPlugin():
    async def setup(self):
        pass
 
    async def run(self, ctx):
        viewer = await api.createWindow(src='https://kaibu.org/#/app')
        #image = np.random.randint(0, 255, [500, 500, 3], dtype='uint8')

        # view image
        #await viewer.view_image(image, type="itk-vtk", name="random pixels")

        #points = np.random.randint(0, 500, [100, 2], dtype='uint16')
        #layer = await viewer.add_points(points, face_color="red")

        # async def say_hello():
        #     await api.alert('Hello!')
        # TODO tasks, add bbox

        async def load_image_from_hpa_umap():
            image_selector = await api.showDialog(src='https://gist.github.com/haoxusci/2cb56343c34dc0057408f6f6edd6ad00')
            image_obj= await image_selector.getImageData()
            self.image_id = image_obj['id'] #str(int(time.time())) # this is the image data folder name, image_id only genreate once when load, to avoid redunctant image saving
            self.image_uri = image_obj['base64']
            api.alert(self.image_id)
            image_selector.hide()
            await viewer.view_image(self.image_uri, type="itk-vtk", name="hpa image")
            #self.layer = await viewer.add_shapes([], shape_type="polygon", edge_color="red", name="geojson")

        #async def load_image():
        #    image_selector = await api.showDialog(src=image_selection_source)
        #    image_url = await image_selector.getImageID()
        #    image_selector.close()
        #    await viewer.view_image(image_url, {'type':"itk-vtk", 'name':"hpa image"})
        def model_train():
            train(batch_size=5,iterations=10)        
        
        def model_pred(image_array):
            # generate a mask file in a tempory folder, assuming hpa_dataset/1600413253/cell_border_mask.png
            # return the cell__border_mask_labels.png, 2d image with each cell as an int number
            #api.alert(image_array.shape)
            img_mask = predict(image_array, size_limit=200)
            api.alert(str(np.unique(img_mask)))
            #labeled_mask = './data/hpa_dataset/tmp_mask/cell_border_mask_labels.png' #this is the prediction func. return a file or numpy array of labeled cells
            #img_mask = imread(labeled_mask)
            return img_mask # 2d values

        async def predict_cell_mask():
            # get the rgb images. currently one issue is we cannot make it run for the mask_to_geojson in this func
            #image_array = imread(BytesIO(base64.b64decode(self.image_uri[self.image_uri.find(",")+1:])))
            image_array = imread(base64.b64decode(self.image_uri[self.image_uri.find(",")+1:]))
            # pass the image_array for prediction
            img_mask = model_pred(image_array)
            ###img_mask = imread('hpa_dataset/temp_mask/cell_border_mask_labels.png')
            #api.alert(img_mask.shape)
            shapes = mask_to_geojson(api, img_mask, label='cell', simplify_tol=1.5)
            #api.log(shapes)
            #return features
            self.geojson_layer = await viewer.add_shapes(shapes, shape_type="polygon", edge_color="red", name="cell")
            #self.layer.set_features(features)

        async def add_new_sample(folder):
            # save image sample, channel image, annotation json, mask png
            # await api.alert(str(self.image_uri))
            imagedata = base64.b64decode(self.image_uri[self.image_uri.find(",")+1:])
            image_dir = os.path.join(DATASET_DIR, folder, self.image_id)
            os.makedirs(image_dir, exist_ok=True)
            # save image data
            with open(os.path.join(image_dir, 'hpa.jpg'), 'wb') as f:
                f.write(imagedata)
            img = imread(os.path.join(image_dir, 'hpa.jpg'))
            imsave(os.path.join(image_dir, 'microtubules.png'), img[:,:,0])
            imsave(os.path.join(image_dir, 'protein.png'), img[:,:,1])
            imsave(os.path.join(image_dir, 'nuclei.png'), img[:,:,2])
            # save intermediate annotation.json
            with open(os.path.join(image_dir, 'annotation.json'), 'w', encoding='utf-8') as f:
                json.dump(await self.geojson_layer.get_features(), f)
            # do a postprocessing for the annotation.json file, and overwrite the file
            file = open(os.path.join(image_dir, 'annotation.json'), "r")
            data = json.load(file)
            data["bbox"]=[0, 0, img.shape[0] - 1, img.shape[1] - 1]
            for item in data['features']:
              item['properties']['label'] = 'cell'
            with open(os.path.join(image_dir, 'annotation.json'), 'w', encoding='utf-8') as f:
                json.dump(data, f)
            # generate the mask file
            files = [os.path.join(image_dir, 'annotation.json')]
            gen_mask_from_geojson(files, masks_to_create_value=['border_mask'])

            ## ToDo

        await viewer.set_ui({"title": "Utilities",
                             "elements": [
                                 {"_rintf": True,
                                   "type": "button",
                                   "label": "Load HPA Image",
                                   "callback": load_image_from_hpa_umap
                                 },
                                 {"_rintf": True,
                                   "type": "button",
                                   "label": "Start Training",
                                   "callback": model_train
                                 },
                                 {"_rintf": True,
                                   "type": "button",
                                   "label": "Evaluate",
                                   "callback": predict_cell_mask
                                 },
                                 {"_rintf": True,
                                   "type": "button",
                                   "label": "Predict",
                                   "callback": predict_cell_mask
                                 },
                                 {"_rintf": True,
                                   "type": "button",
                                   "label": "Send for Training",
                                   "callback": lambda: add_new_sample('train')
                                 },
                                 {"_rintf": True,
                                   "type": "button",
                                   "label": "Send for Evaluation",
                                   "callback": lambda: add_new_sample('test')
                                 },
                                 
                             ]
                             })
api.export(ImJoyPlugin())