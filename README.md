# Co-AMamba
The official repo of the paper Co-AMamba: Co-salient Object Detection with Attention and State Space Model.
 
# Environment Requirement
create enviroment and intall as following: pip install -r requirements.txt
# Data Format
trainset: DUTS+COCO-SEG

testset: CoCA, CoSOD3k, Cosal2015

Co-AMamba

   ├── other codes
   
   ├── ...
   
   ├── datasets
   
         ├── sod
         
              ├── gts
              
                   ├──seg_duts
                   
                        ├──Category 1
                        
                        ├──  ...
                        
                        ├──Category n
                        
                   ├──CoCA
                   
                        ├──Category 1
                        
                        ├──  ...
                        
                        ├──Category n
                        
                   ├──CoSal2015
                   
                        ├──Category 1
                        
                        ├──  ...
                        
                        ├──Category n
                        
                   ├──CoSOD3K
                   
                        ├──Category 1
                        
                        ├──  ...
                        
                        ├──Category n
                        
              ├── images
              
                    ├──seg_duts
                    
                        ├──Category 1
                        
                        ├──  ...
                        
                        ├──Category n
                        
                   ├──CoCA
                   
                        ├──Category 1
                        
                        ├──  ...
                        
                        ├──Category n
                        
                   ├──CoSal2015
                   
                        ├──Category 1
                        
                        ├──  ...
                        
                        ├──Category n
                        
                   ├──CoSOD3K
                   
                        ├──Category 1
                        
                        ├──  ...
                        
                        ├──Category n

# Datasets
Datasets can be downloaded from : 
# Trained model
trained model can be downloaded from :

Run test.py for inference.

The evaluation tool please follow: https://github.com/zzhanghub/eval-co-sod
# Usage
Download pretrainde backbone model :
# Prediction results
The co-saliency maps of Co-AMamba can be found at :
# Contact
Feel free to send e-mails to me (h17630706529@163.com).
