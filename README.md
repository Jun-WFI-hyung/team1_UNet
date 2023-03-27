First 
 - Check and customize Unet_config.json
Second
 - train : python train.py [W or U : Windows or Ubuntu] [T or F : load pth True or False]
   - optional arg: -p unet_epoch000.pth [Put in pth-filename]
   - e.g. python train.py U T -p unet_epoch075.pth

 - test : python inference.py


-- dir structure --<br/>
╔— [Carla_Data] ┬ [rgb] ┬ [train] — *.png<br/>
│               │       ┝ [val] — *.png<br/>
│               │       └ [test] — *.png<br/>
│               │<br/>
│               ┝ [seg] ┬ [train] — *.png<br/>
│               │       ┝ [val] — *.png<br/>
│               │       └ [test] — *.png<br/>
│               │<br/>
│               └ classes.json<br/>
│<br/>
╠— [carla_pth] ┬ [log] *.txt : inference eval log folder [loss, IOU, ..]<br/>
│              └ *.pth : pth save / load folder<br/>
│<br/>
╠— [infer] ┬ [img] — *.png : inference image save folder<br/>
│          └ [log] — *.txt : inference log folder [loss, IOU, ..]<br/>
│<br/>
╠— [net] ┬ Unet.py<br/>
│        └ UnetData.py<br/>
│<br/>
╠— [utils] ┬ accuracy.py<br/>
│          ┝ ImgAug.py<br/>
│          ┝ read_arg.py<br/>
│          ┕ save_load.py<br/>
│<br/>
┝ inference.py<br/>
┝ train.py<br/>
┝ Unet.py<br/>
┝ UnetData_ver_AD.py<br/>
┕ Unet_config.json<br/>
