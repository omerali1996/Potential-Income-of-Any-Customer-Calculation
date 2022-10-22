
#############################################
# Potential Income of Any Customer Calculatıon
#############################################

#############################################
# Problem
#############################################
# One of game sector company aims to calculate potential revenue by separating them exact segments and using some features
#of customers to create level based customer ıdentification.


# For Instance: Company aims to calculate potential revenue of female customer from USA, 25 years old, android user


#############################################
# History of Data
#############################################
# Persona.csv file contains one of gaming industry companies customer ınformation.
# Price: Prıce of transaction
# Source: Which source the customers use
# Sex: Gender
# Country:
# Age:

################# Before  #####################

#    PRICE   SOURCE   SEX COUNTRY  AGE
# 0     39  android  male     bra   17
# 1     39  android  male     bra   17
# 2     49  android  male     bra   17
# 3     29  android  male     tur   17
# 4     49  android  male     tur   17

################# After #####################

#       customers_level_based        PRICE SEGMENT
# 0   BRA_ANDROID_FEMALE_0_18  1139.800000       A
# 1  BRA_ANDROID_FEMALE_19_23  1070.600000       A
# 2  BRA_ANDROID_FEMALE_24_30   508.142857       A
# 3  BRA_ANDROID_FEMALE_31_40   233.166667       C
# 4  BRA_ANDROID_FEMALE_41_66   236.666667       C


#############################################
#############################################
import numpy as np
import pandas as pd
data=pd.read_csv("Cases\persona.csv")
#############################################
#############################################

# Read the persona.csv  and
# show general ınformation about data.
data=pd.read_csv("Cases\persona.csv")
df=data.copy()
df.shape
df.info
df.columns
df.value_counts()
df.describe()
##############Alternative

def check_df(dataframe,head=10):
    print("** First 10 observations:**")
    print(df.head())
    print("** Last 10 observations:**")
    print(df.tail())
    print("** Shape:**")
    print(df.shape)
    print("** Columns:**")
    print(df.columns)
    print("** Types:**")
    print(df.dtypes)
    print("** Numerical Values:**")
    print(df.describe().T)
    print("** Null Values:**")
    if df.isnull().values.any() == True:
        print(df.isnull().sum())
    else:
        print(df.isnull().values.any())
check_df(df)

# How many unique SOURCE inside data? Frequency?

df["SOURCE"].nunique()
df["SOURCE"].value_counts()


# Soru 3: How many unique PRICE?

df["PRICE"].nunique()

# How many sales occured grouping by prices?

df["PRICE"].value_counts()

# How many sales occured grouping by countries?

df["COUNTRY"].value_counts()

# How many total sales occured grouping by countries?
df.groupby("COUNTRY").agg({"PRICE":"sum"})

############Alternative
df.pivot_table("PRICE", ["COUNTRY"], aggfunc="sum")
# How many sales occured grouping by source?

df.groupby("SOURCE").agg({"PRICE":"count"})
df["SOURCE"].value_counts()
# Mean of price grouping by country?

df.groupby("COUNTRY").agg({"PRICE":"mean"})

# Mean of price grouping by source?

df.groupby("SOURCE").agg({"PRICE":"mean"})

# Mean of price grouping by country-source?

df.groupby(["SOURCE","COUNTRY"]).agg({"PRICE":"mean"})

#############################################
# Mean of price grouping by country-source-age-sex?
#############################################

df.groupby(["AGE","SEX","SOURCE","COUNTRY"]).agg({"PRICE":"mean"})

#############################################
# Sort output by price save as agg_df.
#############################################

agg_df=(df.groupby(["AGE","SEX","SOURCE","COUNTRY"])
        .agg({"PRICE":"mean"}))\
    .sort_values(by="PRICE",ascending=False)

#############################################
# Convert indexes as columns.
#############################################
agg_df.reset_index(inplace=True)

#############################################
# Convert  type of age values into categorical and cut reasonable range as agg_cat and insert to agg_df
# Such as: '0_18', '19_23', '24_30', '31_40', '41_70'
#############################################
df["AGE"]=df["AGE"].astype(dtype="category")
agg_df["AGE_CAT"]=pd.cut(agg_df["AGE"],[0,18,23,30,40,66],
                         labels=["0_18","19_23","24_30","31_40","41_70"])

############# With Function

def range_age(x):
    ag_list=[]
    for yas in x:
        if yas> 0 and yas <=18:
            ag_list.append("0_18")
        elif yas > 18 and yas <= 23:
            ag_list.append("19_23")
        elif yas > 23 and yas <= 30:
            ag_list.append("24_30")
        elif yas > 30 and yas <= 40:
            ag_list.append("31_40")
        elif yas > 40 and yas <= 70:
            ag_list.append("41_70")
    return pd.Series(ag_list)

agg_df["AGE_CAT"]=range_age(agg_df["AGE"]).astype(dtype="category")



#############################################
# Designate new level based customer and append to agg_df.
#############################################
# USA_ANDROID_MALE_0_18 that values can be more than into agg_df and need to be singularise, so that values should be
# grouped by mean value of price column
####################################################################
customer_level_based_list=[f"{(agg_df['COUNTRY'][x]).upper()}" \
                           f"_{(agg_df['SOURCE'][x]).upper()}" \
                           f"_{(agg_df['SEX'][x]).upper()}" \
                           f"_{agg_df['AGE_CAT'][x]}"
                           for x in range(len(agg_df.index))]

agg_df.insert(loc=0,column="customer_level_based",value=customer_level_based_list)
new_df=agg_df.groupby(agg_df["customer_level_based"]).agg({"PRICE":"mean"})

############################################################ Alternative1

new_col=[f"{(agg_df['COUNTRY'][x]).upper()}_" \
         f"{(agg_df['SOURCE'][x]).upper()}_" \
         f"{(agg_df['SEX'][x]).upper()}_" \
         f"{agg_df['AGE'][x]}"
         for x in range(len(agg_df.index)) ]

agg_df["NEW_COL"]=new_col

new_df=agg_df.groupby(agg_df["customer_level_based"]).agg({"PRICE":"mean"})


############################################################Alternative2
agg_df["customer_level_based"] = agg_df["COUNTRY"].apply(lambda x : x.upper()) + "" \
                                 + agg_df["SOURCE"].apply(lambda x : x.upper()) + "" \
                                 + agg_df["SEX"].apply(lambda x : x.upper()) + "" \
                                 + agg_df["AGE_CAT"].apply(lambda x : x.upper())
new_df=agg_df.groupby("customer_level_based").agg({"PRICE": "mean"})
new_df.reset_index(inplace=True)

##########################Alternative3

cols = [col for col in agg_df.columns if col not in ["PRICE", "AGE"]]

agg_df["CUSTOMERS_LEVEL_BASED"] = agg_df[cols].apply(lambda x: '_'
                                                     .join(x.values.astype(str))
                                                     .upper(), axis=1)
agg_df = agg_df.groupby(agg_df["CUSTOMERS_LEVEL_BASED"]).agg({"PRICE": "mean"})
agg_df.reset_index(inplace=True)
agg_df.isnull().values.any()

#####################Alternative4
agg_df["COUNTRY"] = np.array(agg_df["COUNTRY"])
agg_df["SOURCE"] = np.array(agg_df["SOURCE"])
agg_df["SEX"] = np.array(agg_df["SEX"])
agg_df["AGE_CAT"] = np.array(agg_df["AGE_CAT"])

agg_df["customers_level_based"] = agg_df["COUNTRY"] +"" + agg_df["SOURCE"] +"" + agg_df["SEX"] +"" + agg_df["AGE_CAT"]

agg_df["customers_level_based"] = [col.upper() for col in agg_df["customers_level_based"]]

#############################################

# Segment new customer such as (USA_ANDROID_MALE_0_18) .
#############################################

###### Different segments methods
"""
agg_df["PRICE"].describe()
def prıce_segment(x):
    seg_list = []
    for price in x:
        if price >= 0 and price <= 15:
            seg_list.append("5.Segment")
        elif price > 15 and price <= 25:
            seg_list.append("4.Segment")
        elif price > 25 and price <= 35:
            seg_list.append("3.Segment")
        elif price > 35 and price <= 45:
            seg_list.append("2.Segment")
        elif price > 45 and price <= 60:
            seg_list.append("1.Segment")
    return pd.Series(seg_list)
agg_df["SEGMENT"]=prıce_segment(agg_df["PRICE"])
yeni_df["SEGMENT"]=prıce_segment(yeni_df["PRICE"])
"""

#######################################################
new_df["SEGMENT"]=pd.qcut(new_df["PRICE"],4,labels=["D","C","B","A"])
new_df.groupby("SEGMENT").agg({"PRICE":["mean","max","sum"]})

new_df.reset_index(inplace=True)
##########################################Alternative1
agg_df["SEGMENT"]=pd.cut(agg_df["PRICE"],[8,32,35,36,60],labels=["D","C","B","A"])

agg_df.groupby("SEGMENT").agg({"PRICE":["mean","max","sum"]})

#############################################
#Estimate new customers' revenue
#############################################

def ıncome(sex,age,source,country,data):
    gender=""
    if sex =="female":
        gender+="female"
    else:
        gender+="male"

    x=""
    if age > 0 and age <= 18:
        x+="0_18"
    elif age > 18 and age <= 23:
        x+="19_23"
    elif age > 23 and age <= 30:
        x+="24_30"
    elif age > 30 and age <= 40:
        x+="31_40"
    elif age > 40 and age <= 70:
        x+="41_70"

    source=source.lower()

    country=country.lower()
    nation=""

    if country=="american":
        nation+="usa"
    elif country=="turkish":
        nation+="tur"
    elif country == "french":
        nation += "fra"
    elif country == "canadian":
        nation += "can"
    elif country == "brazilian":
        nation += "bra"
    elif country == "deusch":
        nation += "deu"

    price = round(data[(data["COUNTRY"] == nation) &
                 (data["SOURCE"] == source) &
                 (data["AGE_CAT"] == x) &
                 (data["SEX"] == gender)]["PRICE"].mean(),3)

    segment=""
    if price >= 8 and price <= 32:
        segment+=("D Segment")
    elif price > 32 and price <= 35:
        segment+=("C Segment")
    elif price > 35 and price <= 36:
        segment+=("B Segment")
    elif price > 36 and price <= 60:
        segment+=("A Segment")

    print(f"{age} years old,  {source.upper()} user  "
          f"{country.capitalize()} {sex},\n"
          f"{segment} and revenue will be {price} .")


# which segment is 33 years old, Android user turkish woman and what is expected revenue?

new_user1="TUR_ANDROID_FEMALE_31_40"
new_df[new_df["customer_level_based"] == new_user1]

###with func
ıncome("female",33,"android","turkish",agg_df)

#which segment is 35 years old,  Ios user french woman and what is expected revenue?

new_user2="FRA_IOS_FEMALE_31_40"
new_df[new_df["customer_level_based"] == new_user2]

### with func
ıncome("female",35,"ios","french",agg_df)


######################################################3
