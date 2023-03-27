First 
 - Check and customize Unet_config.json
Second
 - train : python train.py [W or U : Windows or Ubuntu] [T or F : load pth True or False]
   - optional arg: -p unet_epoch000.pth [Put in pth-filename]
   - e.g. python train.py U T -p unet_epoch075.pth

 - test : python inference.py [W or U : Windows or Ubuntu] [Put in pth-filename]
   - e.g. python inference.py U unet_epoch075.pth
