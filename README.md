## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
  ```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```

<img width="774" height="507" alt="image" src="https://github.com/user-attachments/assets/da77f7d0-062a-4853-98e9-e7dd1757c4cb" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

<img width="558" height="263" alt="image" src="https://github.com/user-attachments/assets/c5da09e3-85ae-4e52-b1d2-7dead31fd89c" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```

<img width="850" height="513" alt="image" src="https://github.com/user-attachments/assets/721f941f-1e9d-429d-a612-a8ae632a8cf6" />


```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```

<img width="554" height="494" alt="image" src="https://github.com/user-attachments/assets/54ed9066-585a-4095-88c4-1cf4df42c360" />

```
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

one = OneHotEncoder(sparse_output=False)   # <-- use sparse_output
df2 = df.copy()
enc = pd.DataFrame(one.fit_transform(df2[["nom_0"]]))
df2 = pd.concat([df2, enc], axis=1)
df2
```


<img width="669" height="496" alt="image" src="https://github.com/user-attachments/assets/081f01d4-1e2a-46fd-9924-e372ad30c566" />

```
pd.get_dummies(df2,columns=["nom_0"])
```

<img width="873" height="501" alt="image" src="https://github.com/user-attachments/assets/43013111-1acb-436e-b7d5-c0fae5930f8f" />

```
pip install --upgrade category_encoders
```


<img width="1736" height="464" alt="image" src="https://github.com/user-attachments/assets/58ea3c00-6f71-4029-a961-0fb6a68746f4" />

```
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```


<img width="839" height="499" alt="image" src="https://github.com/user-attachments/assets/d7a74abc-ebde-4247-a4f7-d1de01ece01d" />

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```


<img width="644" height="504" alt="image" src="https://github.com/user-attachments/assets/09059d14-5a2a-456a-937f-e4f4e0f0fb23" />

```
dfb=pd.concat([df,nd],axis=1)
dfb
```


<img width="973" height="484" alt="image" src="https://github.com/user-attachments/assets/b2646ffe-a51f-44e4-9888-d70dbe4f2140" />

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```


<img width="825" height="498" alt="image" src="https://github.com/user-attachments/assets/d1bef5c5-75e3-40aa-bfad-3adfa6d83e73" />

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```


<img width="1145" height="585" alt="image" src="https://github.com/user-attachments/assets/88f38879-818a-4d18-a3c3-72925cb22e22" />

```
df.skew()
```


<img width="481" height="290" alt="image" src="https://github.com/user-attachments/assets/3b8acc39-7bac-4445-80f7-fb5d72ddbbcf" />

```
np.log(df["Highly Positive Skew"])
```


<img width="379" height="622" alt="image" src="https://github.com/user-attachments/assets/01f620c1-d8f7-4e57-8ad4-0a41637e84cd" />

```
np.sqrt(df["Highly Positive Skew"])
```


<img width="427" height="638" alt="image" src="https://github.com/user-attachments/assets/7e29aa8a-246e-4056-b8be-4abd6ce4f2be" />

```
np.reciprocal(df["Moderate Positive Skew"])
```


<img width="482" height="635" alt="image" src="https://github.com/user-attachments/assets/8dee2475-4496-48a4-8303-5e2e8fb87f90" />

```
np.square(df["Highly Positive Skew"])
```


<img width="433" height="643" alt="image" src="https://github.com/user-attachments/assets/e246e8f5-165f-4998-98ce-b1c64b142d75" />

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```


<img width="1470" height="593" alt="image" src="https://github.com/user-attachments/assets/cb82ac62-3e8a-4c9b-9e72-bdae1aea8886" />

```
df.skew()
```


<img width="506" height="328" alt="image" src="https://github.com/user-attachments/assets/11ff21f4-525f-4323-b723-b05d6bd7f388" />

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```


<img width="597" height="381" alt="image" src="https://github.com/user-attachments/assets/68c6838c-abdf-4ade-89c8-3da85dd2be35" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```


<img width="1657" height="638" alt="image" src="https://github.com/user-attachments/assets/fdb6cd6d-4a6b-4d7a-b41c-5c188c2f5469" />


```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

<img width="840" height="612" alt="image" src="https://github.com/user-attachments/assets/7113041c-3233-41f7-bf94-98e07d038a4b" />

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```


<img width="849" height="608" alt="image" src="https://github.com/user-attachments/assets/94e5a94a-ef48-4e03-81f1-6c5392a4d123" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```


<img width="901" height="597" alt="image" src="https://github.com/user-attachments/assets/7d759503-272d-4900-8d81-f949dc69d7f1" />

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```


<img width="877" height="638" alt="image" src="https://github.com/user-attachments/assets/50fef8a4-ed27-4f65-b251-a731034496c2" />

```
dt=pd.read_csv("data.csv")
dt
```

<img width="645" height="495" alt="image" src="https://github.com/user-attachments/assets/38176107-3e1a-4556-9289-77903d88ab9f" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Ord_1"]=qt.fit_transform(dt[["Target"]])
sm.qqplot(dt['Target'],line='45')
plt.show()
```



<img width="1122" height="690" alt="image" src="https://github.com/user-attachments/assets/444ee32a-2041-4b36-8254-7f42d38abe84" />

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```


<img width="935" height="622" alt="image" src="https://github.com/user-attachments/assets/25ead79e-e9d3-450f-9efe-669c7192dcdc" />






















# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file
was performed successfully.
       

       
