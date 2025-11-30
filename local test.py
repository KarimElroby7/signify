from ultralytics import YOLO
#dataset_path =  r"C:\Users\karimelroby\PycharmProjects\PythonProject18\ag\data.yaml"
#model = YOLO("yolo11n.pt") # load the model

model = YOLO("./models/best.pt") # load the model
#results = model.train(data=dataset_path, epochs=30)

#model.save(r".\models\hos.pt")
results = model("./g1.jpg",show=True,save=True)


#results = model(source=0,show=True,save=True,conf=0.5)
#results = model(r"C:\Users\karimelroby\Pictures\Camera Roll\1.mp4",show=True,save=True,conf=0.5)






