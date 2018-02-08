# SRGAN
Implementation of SRGAN + SRResnet in PyTorch

WIP effort to get a solid pytorch implementation up and running. Tensorflow isn't transparent enough, so I want to have an 
alternative super-res project avilible. 


#### status: SRResNet pretraining with MSE works. Still working on kinks in SRGAN


### SRResNet Results (200k iterations, batch size: 4)

Mean MSE: 0.011

Mean PSNR: 20.65

#### Low Res - High Res - SR Low Res
![plot1](results/srresnet/plot1.png)
![plot2](results/srresnet/plot2.png)
![plot3](results/srresnet/plot3.png)

