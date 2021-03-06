# UROP-Machine-Learning-project
This is a repository to store my codes for my UROP project in AY2019-20, under the supervision of Prof Shen Lei

The aim of this project is to use Machine Learning algorithm to predict the bulk modulus of Heusler Alloys, a family of alloys with similar chemical composition and structure. 
All the data is collected through online database, not via actual laboratory work. 

Some literature work on Machine Learning application in material science have been referenced, and most of them take rather sophisticated approaches towards their problem. However, given limited experience and time, as well as resources, I as an undergraduate was not able to replicate their methods in this tiny project. That been said, they do help me in identifying some candidates in my choice of Machine Learning algorithm. 

I have written the codes myself, with reference to an example which is provided by the following link:https://nbviewer.jupyter.org/github/hackingmaterials/matminer_examples/blob/master/matminer_examples/machine_learning-nb/bulk_modulus.ipynb. The overall process of my project follows his quite closely, but since we are dealing with different datasources, there was some heavylifting on data prepocessing that has to be done entirely on my own. Moreover, due to a rather small sample size, I was unable to reach the similar level of accuracy of this example does, so I have to analyze the model performance in a more detailed manner, which is not covered in this example. 

The idea of using learning curves and hyper-parameter tunning are learnt from the book Python Machine Learning third edition by Sebastian Raschka & Vahid Mirjalili. 
You could see my analysis in the project report I have uploaded. 

I have included both codes and the Jupyter notebook files in the repository. Since Github does not render .ipynb files nicely, you could view them for more colorful illustration using nbviewer here: https://nbviewer.jupyter.org/.
