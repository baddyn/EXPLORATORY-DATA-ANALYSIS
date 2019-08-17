
# coding: utf-8

# In[3]:


#eda is exploratory data analysis 
#using linear algebra or using simle tools like plotting ,statistics 
#to understand about our dataset


#iris flower data set analysis 
#a garden has 3 types of flowers 
#someone gives us a flower and to tell which one of these 3 ia this one

#simple classification problem
#given 4 features
#sepal length,sepal width,petal legth,petal width and based on these 
#classified the flower into 3 types

#sepal is larger one and petal smaller one


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

iris=pd.read_csv('iris.csv')


# In[5]:


iris.shape


# In[6]:


iris.head


# In[7]:


iris.columns


# In[8]:


#we have to predict variety given other 4 columns

#to check how many rows belong to which class
iris["variety"].value_counts()


# In[9]:


#this is a balanced dataset
#as all variety are almost equally probable
#DEPENDING ON THE BALANCED OR UNBAL NATURE
#DIFFERENT TYPE OF DATA ANALYSIS IS TO BE DONE


# In[10]:


#LEARNING 2D SCATTER PLOTS
#IMPORTANT

#always make note of starting points of axes,the scale and what the axes represent
iris.plot(kind='scatter',x='sepal.length',y='sepal.width');
plt.show()


# In[11]:


#here the strting point is not 0,0
#it is 4.5,2
#here we have plotted sepal length and sepal width

#this tells the variation of sepal lenth and width
#but in this we arent able to detect which one is which variety

#so color based on class which they belong to

#we will use seaborn for thsi

sns.set_style('whitegrid');
#hue defines basis for coloring

sns.FacetGrid(iris,hue="variety",size=6).map(plt.scatter,"sepal.length","sepal.width").add_legend(); 
plt.show();


# In[12]:


#legend is the color scale that is shown on rhs

#from above plot see that orange and green are mixed but blue is seperate 
#so draw aline seperating lue and other 2
#after drawingline  one point  is left 
#this means that oly these 2 features can distinguish one specis from other

#the concept used is linear seperation

#but this cant be used to distinguish bw ither 2

#the qn is why arent we taking 3d scatter plots inc all feature
#we can draw them and now we ll draw them

#but problem is 4d cant be visulaised 
#3d also requires lot of mouse moveemnts

#so 3d plot is not preferred


# In[13]:


#3d scatter plots
#plotly is agood website to create 3d scatter plots
#as it is interactive

#https://plot.ly/pandas/3d-scatter-plots/ 
#visit this link

#but less preferred

# in 4d prob is we cant see
#so we use a trick to visulaise it


# In[14]:


#pair plot is the trick

#try to crete pairs of 2 to visuslaise for any higer dimension

#here 4c2 combinations possible
#6 combinations

#again ue seaborn

plt.close();
sns.set_style("whitegrid");
sns.pairplot(iris,hue="variety",size=3);
plt.show();


# In[15]:


#note these plos are kind of a matrix
#in 1st row the y axis is sepal length as marked
#in first column sepal length as marked

#but we said only 6 possible but here are 16 plots 
#out of which 4 diagonal are useleess
#as when x and y areboth sepal length it is useless

#also the plots below diag are same as plots above just with 
# x and y revresed


#now analyse
#now 1,2 and 1,3 and 1,4 sepearte blues
#as we can see

#1,3 and 1,4 are better seperation

#so lets look at this values
#i.e. look at 3,4
#this perfectly seperates blue 

#now after looking at this pair plot 3,4
#lets build a model 
#using if else


#  if(pl<=2) and pw<=1  
#  category 1 i.e. setosa

#now for other two categories
#look carefully in the 3,4 itself 
#roughly we can say

#if pw >1 and pw<2 and pl<5 and pl>3 
#bascially the box of category 2 is taken
#then it is category 2
#now there is some error
#which willl be there

#we cant make 100% correct approximation


#so we can say petal ength and petal width are most important features


# In[16]:


#limitations of pair plots

#for 100 features we will have 100c2 
#which are impossible to handle

#techni

#so pair plots can be handled upto max 10 dimensions

#now we have talked about 2d,3d etc scattter plots

#what about 1d plots
#lets look at them

#points starts overlapping 
#we cant distingish beteen them
#as they all lie on same straight line

#to prevent this

#we break interval into parts say 1-2 in 5 parts and the then the 
#interval 1-1.2 we will store also the count in that interval

#this is a histogram
import seaborn as sns
sns.FacetGrid(iris,hue='variety',size=5).map(sns.distplot,'petal.length').add_legend();
plt.show();


# In[17]:


#these are histograms
#orange is hsitogram of petal lenth of setosa
#y axis contains counts 

#the smooth lines or curveis probablilty density function
#pdf=smooth histogram
#kernel densty estimation

#interpretation

#no stosa for petal_length greter than 2
# if pl<=2 setosa
# else not

# now versicolr and virginica pdfs are overlappping
# there is overlapping region

#we will have to make mistake here

#if pl<2 setosa
# if pl<4.7 versicor
#else virginica

#other thn 4.7 we could have choosen 5 but it will have more mistakes
#bcoz for pl 5 htere are more virginica
#at 4.5 versicolor wins
#there definitely has to be errors
#but to reduce error we havemade our choice

#this is also called a density plot
# bcoz it tells abt the density of the region


# In[18]:


#univariate analysis 

#one variable analysis
#say we have to chosose out of 4 var which of them is more useful 
#we can answer these using pdf etc.

#draw histograms for all 4 variables


# In[19]:


sns.FacetGrid(iris,hue='variety',size=5).map(sns.distplot,'petal.width').add_legend();
plt.show();


# In[20]:


sns.FacetGrid(iris,hue='variety',size=5).map(sns.distplot,'sepal.length').add_legend();
plt.show();


# In[21]:


sns.FacetGrid(iris,hue='variety',size=5).map(sns.distplot,'sepal.width').add_legend();
plt.show();


# In[22]:


#now we want he pdfs to be more seperated as it will make prediction easier

#after seeing pdfs  roughly for pl,pw 
#pl better coz atleast setosa seperated and other 2 almost same in both

#sl is disaterous
#all merged

#so till now pl is the best

#sw again very bad
#so pl wins
# pl>pw>>sl>>sw
    


# In[23]:


# cdf  cumultative density fucntion
#cdf tells how many points have values less than or equal to a particular values

#note to plot pdf,cdf we can also use probab. on y axis

#cdf starts from 0 and ends at 1 for proabab graph

#cdf is the sum of pdfs till now 
#cumulative sum of pdfs till now
# area under pdf curve is cdf
#so if differentiate cdf u get pdf
#integrate pdf u get cdf


# In[24]:


#plotting cdf
import numpy as np
counts,bin_edges=np.histogram(iris['petal.length'],bins=10,density=True)
pdf=counts/(sum(counts))
print(pdf)

print(bin_edges)

#computing cdf
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

plt.show();


# In[25]:


# by using cdf we can tell that say for values 4.7 
#versicolor ha cdf 90% then 90% of versicolor are coreectly marked
#so i am makig 10% mistake in versicolor

#this info isvery usefula nd can only be obtained fro cdf


# In[26]:


# mean variance and stdd dev

#say we are given setosa petal length for 50 
#w eknow mean is av.
print(np.mean(iris['petal.length']))


# In[29]:


#this is avg petal length

#we can plot this seperately for 3 species and see which has higher values

#problem with mean
#say someone took another reading and then mistakenly entered wrong values
#and entered pl as 100  this is abnoraml and also called outliner
#this clearly changes mean by a large value
#that is due to this number mean changes abruptly

#mean tells abt cental tendency
#its avg.


#note from the pdfs above

#the width for setoa is less for pdfs
#for others its more
#this is also called spread
#for virg. and vers. spread is more that points are more spreaded
#that is more points in setosa are ear the mean not in other 2

#note this spread is our VARIANCE
#SPREAD BASICALLY MEANS HOW FAR ARE THE POINTS FROM MEAN(SQUARED DISTANCES) 
#SUMMATION OF SQAURES OFABOVE QUANT FOR ALL POINTS DIVIDED BY N GIVES VARIANCE
#VARIANCE IS AVG OF SQAURED DISTANCES

#SQRT(VAR)=STANDARD DEVAIATION

#SPREAD IS MORE PRECISELY STDD DEVIATION AND NOT VARIANCE

print(np.std(iris['petal.length']))
#std deviation for whole iris flowers


#just like mean varinace ans=d stdd deviation can be corrupted by one corrupt value
#or one outliner

# so for non corruptance


# In[31]:


#median percentile quantile iqr mad

print(np.median(iris['petal.length']))
#will remain almost same even with outliner
#small no. of outliners cant corrupt the data
#if less than50% points are corrupt median doesnt get corrupt


#50th percentile value is median

#25,50,75 ,100 are quantiles 
# 25 means 25% values are less than this in set

#quantiles
print(np.percentile(iris['petal.length'],np.arange(0,100,25)))
#gies 0,25,50,75

#similarly we can obain any value by
print(np.percentile(iris['petal.length'],90))
#this is 90th percentile

#percentile values are very useful like jee  mains etc.


# In[32]:


#median absolute deviation

#given a bunch of obsn  median((abs(xi-median)) for i =1 to n)
#not readily availavle in numpy

#median is equivalent to mean except that it is not prone to outliners

#for mad

from statsmodels import robust
print("mad")
print(robust.mad(iris['petal.length']))

#mad is equivalent to stdd deviation except that it is not prone to outlners

#there is one more quant

#iqr
#inter quantile range 
#75th qunatile - 25th quantile(this contains 50% of data)
#no problem of outliner


# In[34]:


#box plot with whiskers

#summary till now

#from pdf we cant get quantiles values directly

#box plot is very important

sns.boxplot(x='variety',y='petal.length',data=iris)
plt.show()


# In[35]:


#lets understand box plot for versicolor

#we have taken petal length for eversicolr
# now for box 
#box has three lines
#line 1 25 quantile,line 2 50,line 3 75

#two lines at start and end are called whsikers 
#whsikers dont generally follow standard procedure
#different ways of defining whsikers

#seaborn uses complex mechanism 1.5*iqr

#we can see that if i take pl =5 as versicolr 
#than virginica willl have 25% wrong detection


# In[37]:


#violin plots

#combination of box plot and pdf types

sns.violinplot(x='variety',y='petal.length',data=iris,size=8)
plt.show()


# In[38]:


#interpretation

#these loook like violin
# the thick black area is box plots box 
#white dot being 50th p,and the two
# lines at tp and bottom are whsikers
#now the plots if we see are pdf and cdf are rotated by 90degreee


# In[39]:


#note all the pdfs , box plpts etc are univariate as only one variable used
#pair plots etc are bivariate
#3d scatter plots are multivariate


# In[40]:


#multivariate probability density,contour plot

#2d density plot,contors-plot
sns.jointplot(x='petal.length',y='petal.width',data=iris,kind='kde')
plt.show();


# In[41]:


#dark color means more points
#imagine as hill coming towards u out of screen
#height greater where color dark

#this is called a comtoir density plot
#all points on the same ellipse have same height

#two plots on top and rhs are univariate plots for pl,pw


#so
#1d density is pdf
#2d density contour

#3d etc dealt later as we need d visualisation to plot 3d


# In[ ]:


#solve haberman cancer survival forom kaggle

#tasks
#erform eda
#1. explain objective
# 2. high level stats  no. of points,size,features etc.
#3 perform univar aalysis
#perform #bivariate analysis
#write obsn in english clearly

