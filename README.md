First 
 - Check and customize Unet_config.json
Second
 - train : python train.py [W or U : Windows or Ubuntu] [T or F : load pth True or False]
   - optional arg: -p unet_epoch000.pth [Put in pth-filename]
   - e.g. python train.py U T -p unet_epoch075.pth

 - test : python inference.py


-- dir structure --
╔— [Carla_Data] ┬ [rgb] ┬ [train] — *.png
│               │       ┝ [val] — *.png
│               │       └ [test] — *.png
│               │
│               ┝ [seg] ┬ [train] — *.png
│               │       ┝ [val] — *.png
│               │       └ [test] — *.png
│               │
│               └ classes.json
│
╠— [carla_pth] ┬ [log] *.txt : inference eval log folder [loss, IOU, ..]
│              └ *.pth : pth save / load folder
│
╠— [infer] ┬ [img] — *.png : inference image save folder
│          └ [log] — *.txt : inference log folder [loss, IOU, ..]
│
╠— [net] ┬ Unet.py
│        └ UnetData.py
│
╠— [utils] ┬ accuracy.py
│          ┝ ImgAug.py
│          ┝ read_arg.py
│          ┕ save_load.py
│
┝ inference.py
┝ train.py
┝ Unet.py
┝ UnetData_ver_AD.py
┕ Unet_config.json