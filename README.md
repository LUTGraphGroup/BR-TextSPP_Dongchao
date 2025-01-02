# BR-TextPP
==
## "Improved automatic ICD coding based on fusion of multi-scale features and fine-tuned pre-training models"
### How to run code?  
Step1:
------
Use the following statement to obtain the preprocessed file data.csv .
			mimic = MIMIC_new(path='path to mimic3')
			mimic.get_basic_data(outPath='data.csv')
```
Step2:
------
Use the following statement to Instance the data utils class and get the pretrained word embedding.
			dataClass = (dataPath, mimicPath='mimic/', stopWordPath="stopwords.txt", validSize=0.2, testSize=0.0, minCount=10, noteMaxLen=768, seed=9527, topICD=-1)   
  `topICD=-1` indicates the MIMIC-III FULL dataset, while `topICD=-1` also indicates the MIMIC-III top 50 dataset.
```
Step3：
------
Use the following statement to compute ICD vectors.
labDescVec = get_ICD_vectors(dataClass=dataClass, mimicPath="path to mimic3")
if dataClass.classNum=50:
    labDescVec = labDescVec[dataClass.icdIndex,:]
```
Step4:
------
Use the following statement to train model.
		model.train(dataClass, trainSize=64 batchSize=64, epoch=100,
            lr=0.003, stopRounds=-1, threshold=0.5, earlyStop=10,
            savePath='model/BR-TextSPP', metrics="MiF", report=["MiF","MaF", "MiAUC","MaAUC","P@8"])
```
   
