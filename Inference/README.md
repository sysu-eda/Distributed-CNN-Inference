


# DeCNN

This implements inference of ResNet-50 using MoDNN[1], Coates et al.[2] and our DeCNN methods.


## Requirements

* ARM Compute Library 19.08
* MPI

## Run
- make
- If not with shared storage, you should:
./deply.sh 
or manually send the compiled file into each device.
- run the file in all devices simultaneously. Write a new script or just use the Xshell to send the command simultaneously.



[1] J.  Mao  and  X.  Chen  and  et  al., MoDNN:  Local  distributed  mobile computing  system  for  Deep Neural  Network ,  in  Design,  Automation Test in Europe Conference (DATE), 2017.

[2] Adam Coates and Brody Huval and et al., Deep learning with COTS HPC  systems ,  in  International  Conference  on  Machine  Learning (ICML), 2013