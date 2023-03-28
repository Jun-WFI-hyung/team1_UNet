 - Check and customize Unet_config.json</br>
 - train : python train.py [T or F : load pth True or False]
   - optional arg: -p unet_epoch000.pth [Put in pth-filename]
   - e.g. python train.py T -p unet_epoch075.pth

 - test : python inference.py [Put in pth-filename]
   - e.g. python inference.py unet_epoch075.pth
