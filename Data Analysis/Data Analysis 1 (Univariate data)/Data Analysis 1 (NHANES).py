import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
import numpy as np
from tabulate import tabulate

da = pd.read_csv("NHANES.csv")
#print(da)
#print(da.columns)

#Give labels to the marital statuses, by creating a new (mirror) distribution called DMDMARTLx, using the replacement rule r
r1 = {1: "Married", 2: "Widowed", 3: "Divorced", 4: "Separated", 5: "NeverMarried", 6: "Cohabitating", 77: "Refused", 99: "Unknown"}
da["DMDMARTLx"] = da["DMDMARTL"].replace(r1)
#print(da.loc[:,'DMDMARTLx'])
r2 = {1: "Female", 2: "Male"}
da["RIAGENDRx"] = da["RIAGENDR"].replace(r2)
#print(da.loc[:,'RIAGENDRx'])

#Prints the percentages of each marital status in out data frame
x = da["DMDMARTLx"].value_counts()
#print(x / x.sum())

#We print out each gender with the percentage frequency; mind there is 2 entries for RIAGENDRx, meaning 2=males and 1=females
for ky,db in da.groupby("RIAGENDRx"):
    #print("\nGender =", ky)
    x = db["DMDMARTLx"].value_counts()
    x = x / x.sum()
    #print(x)

#We take a portion of the population between 30 and 40yo and redo the same analysis of marital status both for men and women separately
da3040 = da.query('RIDAGEYR >= 30 & RIDAGEYR <= 40')
#The .query() method is a powerful way to filter DataFrames using string expressions, similar to SQL WHERE clauses.
# It provides a more readable and concise syntax for selecting rows based on conditions.
for ky,db in da3040.groupby("RIAGENDRx"):
    #print("\nGender =", ky, "and 30 <= age <= 40")
    x = db["DMDMARTLx"].value_counts()
    #print(x / x.sum())

#Restricting to the female population, stratify the subjects into age bands no wider than ten years, and construct the distribution
# of marital status within each age band. Within each age band, present the distribution in terms of proportions that must sum to 1.
#print(da['DMDMARTLx'])
#print(da['RIDAGEYR'])
#print("Table for females")
female_data = da[da['RIAGENDRx'] == 'Female'].copy() #create a "new" database with only the female data, the copy avoids a warning
female_data["ageBands"] = pd.cut(female_data.RIDAGEYR, [18, 20, 30, 40, 50, 60, 70, 80]) #we create the age bands
ex1a = female_data.groupby('ageBands', observed=True)['DMDMARTLx'] #observed True will avoid a warning about NaN data
ex1a = ex1a.value_counts()
ex1a = ex1a.unstack() #To create a table where Marital status is the columns and ageBands is the label
ex1a = ex1a.apply(lambda x : x / x.sum(), axis = 1) #create the table as seen in pag8, axis = 1 means to apply the operation
                                                                         # to each row, unlike columns which arent the ones we want
#print(ex1.to_string(float_format = "%.3f")) would be the raw print, but it looks off, we can use tabulate
#print(tabulate(ex1a.reset_index(), headers='keys', tablefmt='fancy_grid', showindex=False, floatfmt=".3f"))
    #where ex1.reset_index() converts the DataFrame index into regular columns
    #headers='keys' specifies what to use as column headers and it uses the DataFrame's column names as the table header
    #tablefmt='grid' defines the table format/style
    #showindex=False controls whether to show the row index and when set to False, hides the default numeric index column

#We, then, repeat for males
#print("Table for males")
male_data = da[da['RIAGENDRx'] == 'Male'].copy()
male_data["ageBands"] = pd.cut(male_data.RIDAGEYR, [18, 20, 30, 40, 50, 60, 70, 80])
ex1b = male_data.groupby('ageBands', observed=True)['DMDMARTLx']
ex1b = ex1b.value_counts()
ex1b = ex1b.unstack()
ex1b = ex1b.apply(lambda x : x / x.sum(), axis = 1)
#print(tabulate(ex1b.reset_index(), headers='keys', tablefmt='grid', showindex=False, floatfmt=".3f"))

#Construct a histogram of the distribution of heights using the BMXHT variable in the NHANES sample.
ex2a = sns.histplot(da['BMXHT'], kde = True).set_title("Histogram of the distribution of heights")
plt.savefig('height_distribution_v1.png')
#plt.show() #Print the previous plot
plt.close() # Close the figure to free up memory and don't make graphs overlap

#Use the bins argument to histplot to produce histograms with different numbers of bins.
# Assess whether the default value for this argument gives a meaningful result, and comment on what
#  happens as the number of bins grows excessively large or excessively small.
for i in range(1, 12):  # i from 1 to 12, with 2 jump
    sns.histplot(da['BMXHT'], kde=True, bins=3*i).set_title("Histogram of the distribution of heights 2")
    plt.savefig(f'height_distribution_v2_{i}.png') #Using f-strings to properly include the integer in the filename
    plt.close() # Close the figure to free up memory and don't make graphs overlap
#plt.tight_layout()  # This helps to avoid overlapping of titles and labels

#Make separate histograms for the heights of women and men.
ex3 = sns.FacetGrid(da,col="RIAGENDRx")
ex3 = ex3.map(plt.hist, 'BMXHT')
ex3.set_axis_labels("Height (cm)", "Frequency")
plt.tight_layout()  # This helps to avoid overlapping of titles and labels
plt.savefig('Height histograms between men and women.png')
plt.close() # Close the figure to free up memory and don't make graphs overlap

#Make a side-by-side boxplot showing the heights of women and men.
ex4 = sns.boxplot(da, x = da['BMXHT'], y = da['RIAGENDRx'],)
plt.tight_layout()  # This helps to avoid overlapping of titles and labels
plt.savefig('Height boxplots between men and women.png')
plt.close() # Close the figure to free up memory and don't make graphs overlap

#Make a boxplot showing the distribution of within-subject differences between
#  the first and second systolic blood pressure measurents (BPXSY1 and BPXSY2).
plt.subplot(2,1,1)
sns.boxplot(da, x=da['BPXSY1'])
plt.subplot(2,1,2)
sns.boxplot(da, x=da['BPXSY2'])
plt.tight_layout()  # This helps to avoid overlapping of titles and labels
plt.savefig('Boxplots of the blood measurements.png')
plt.close() # Close the figure to free up memory and don't make graphs overlap

#Construct a frequency table of household sizes (DMDHHSIZ) for people within
# each educational attainment category (the relevant variable is DMDEDUC2).
#  Convert the frequencies to proportions. (frequency=count)
r3 = {1: "<9", 2: "9-11", 3: "HS/GED", 4: "Some college/AA", 5: "College", 7: "Refused", 9: "Don't know"}
da["DMDEDUC2x"] = da["DMDEDUC2"].replace(r3)
ex5 = da.DMDHHSIZ.value_counts().reset_index() #reset_index() converts it into a DataFrame
ex5.columns = ['Household size', 'Count']
ex5['Frequency'] = ex5['Count']/ex5['Count'].sum()
#print(tabulate(ex5, headers='keys', tablefmt='grid', showindex=False, floatfmt=".3f"))
     #this last floatfmt=("", "", ".3f")) makes so that the first 2 columns have no floating numbers and the last has 3
education_levels = sorted(da['DMDEDUC2x'].dropna().unique().tolist())
#where: .unique() returns an array of all values found in our column DMDEDUC2x and eliminates duplicates;
# dropna() removes empty values (which could create errors in this case), .tolist() transforms the NumPy array
#  into a standard Python list and sorted sorts the list of unique values in ascending order
#   (alphabetically for strings, numerically for numbers)
#print(education_levels)
for item in education_levels:
    item_data = da[da['DMDEDUC2x'] == item].copy()
    ex6 = item_data.DMDHHSIZ.value_counts().reset_index()  # reset_index() converts it into a DataFrame
    ex6.columns = ['Household size', 'Count']
    ex6['Frequency'] = ex6['Count'] / ex6['Count'].sum()
    ex6['Frequency'] = ex6['Frequency'].apply(lambda x: f"{x:.3f}") #Format only the Frequency column to 3 decimal places
     #lambda x: This is a lambda function (anonymous function) that takes one parameter x (each value in the column)
    # f"{x:.3f}" is an f-string that formats the number x to have exactly 3 decimal places
    #print(f"\nHousehold size distribution for {item}:") #the starting f allows us to use {item} as the actual for loop name thing
    #print(tabulate(ex6, headers='keys', tablefmt='grid', showindex=False))

#Restrict the sample to people between 30 and 40 years of age.
# Then calculate the median household size for women and men within each level of educational attainment.
#print("Data for people between 30 and 40")
#da3040 = da.query('RIDAGEYR >= 30 & RIDAGEYR <= 40')
#print(da3040)
#print(da3040.loc[:,'RIDAGEYR'])
#From now on, for a few lines, we rename the variables with the appropriate string names, for visualization
da3040["DMDEDUC2x"] = da3040["DMDEDUC2"].replace(r3)
r2 = {2: "Female", 1: "Male"}
da3040["RIAGENDRx"] = da3040["RIAGENDR"].replace(r2)
education_levels = sorted(da3040['DMDEDUC2x'].dropna().unique().tolist())  #creates a list with the unique values RIAGENDR assumes
genders =  sorted(da3040['RIAGENDRx'].dropna().unique().tolist())
#print("Median household size for women and men within each level of educational attainment")
for item in education_levels:
    for gender in genders:
        item_data = da3040.query('DMDEDUC2x == @item and RIAGENDRx == @gender').copy()
        median = item_data['DMDHHSIZ'].median()
        #print("The median of the household size for the gender", gender, "and education level", item, "is:", median)
#print("--------------------------------------")
#We could do the same computation by dropping the results for refused and unknown
#print("Data for people between 30 and 40, without marginal values (unknown and refused education levels.")
#da3040 = da.query('RIDAGEYR >= 30 & RIDAGEYR <= 40')
#print(da3040)
#print(da3040.loc[:,'RIDAGEYR'])
#da3040["DMDEDUC2x"] = da3040["DMDEDUC2"].replace(r3)
#education_levels = sorted(da3040['DMDEDUC2x'].dropna().unique().tolist())
#genders =  sorted(da3040['RIAGENDRx'].dropna().unique().tolist())
#r2 = {1: "Female", 2: "Male"}
da3040 = da3040.loc[~da3040.DMDEDUC2x.isin(['Refused', 'Unknown']),:]
#da3040["RIAGENDRx"] = da3040["RIAGENDR"].replace(r2)
#print("Median household size for women and men within each level of educational attainment, in this case")
for item in education_levels:
    for gender in genders:
        item_data = da3040.query('DMDEDUC2x == @item and RIAGENDRx == @gender').copy()
        median = item_data['DMDHHSIZ'].median()
        #print("The median of the household size for the gender", gender, "and education level", item, "is:", median)

#Alternatively, in short, removing the for double cycle
result = da3040.groupby(["RIAGENDRx", "DMDEDUC2x"])["DMDHHSIZ"].median()
#print(result)


#The participants can be clustered into "masked variance units" (MVU) based on every combination of the variables
# SDMVSTRA and SDMVPSU. Calculate the mean age (RIDAGEYR), height (BMXHT), and BMI (BMXBMI) for each
# gender (RIAGENDR), within each MVU, and report the ratio between the largest and smallest mean (e.g. for height) across the MVUs.
#I want to make groups between each combination of the MVU
#print(da['SDMVSTRA'].describe)
#print(da['SDMVSTRA'].value_counts())
age_mean = da.groupby(["SDMVSTRA", "SDMVPSU"])["RIDAGEYR"].mean()
height_mean = da.groupby(["SDMVSTRA", "SDMVPSU"])["BMXHT"].mean()
BMI_mean = da.groupby(["SDMVSTRA", "SDMVPSU"])["BMXBMI"].mean()
#print("age mean: \n", age_mean, "\n," "height mean: \n ",  height_mean, "\n", "BMI mean: \n ",BMI_mean)
MVU1_values_list = sorted(da['SDMVSTRA'].dropna().unique().tolist())  #creates a list with the unique values RIAGENDR assumes
MVU2_values_list =  sorted(da['SDMVPSU'].dropna().unique().tolist())
#print(MVU2_values_list, MVU1_values_list)
three_items = ["RIDAGEYR", "BMXHT", "BMXBMI"]
for variable in three_items:
    lowest = float('inf') #infinty
    highest = float('-inf')
    #print("We now search the smallest and highest mean for", variable)
    for value1 in MVU1_values_list:
        for value2 in MVU2_values_list:
            data_pool = da.query('SDMVSTRA == @value1 and SDMVPSU == @value2').copy()
            mean = data_pool[variable].mean()
            if mean < lowest:
                lowest = mean
                rounded_lowest = round(lowest, 3)
            if mean > highest: #not elif because i may exit the loop for good, using the first if, if they are in order
                highest = mean
                rounded_highest = round(highest, 3)
            #print("The highest and lowest means in", variable, "in the section SDMVSTRA =",
                  #value1, "and SDMVPSU =", value2, "are:", highest, lowest)


#Calculate the inter-quartile range (IQR) for age, height, and BMI for each gender and each MVU.
# Report the ratio between the largest and smalles IQR across the MVUs (between all of them, just section by gender).
#We basically need to do the same but with the interquartile range
print("Females case")
daF = da.query('RIAGENDRx == "Female"').copy()
age_IQR = daF.groupby(["SDMVSTRA", "SDMVPSU"])["RIDAGEYR"].quantile(0.75) - da.groupby(["SDMVSTRA", "SDMVPSU"])["RIDAGEYR"].quantile(0.25)
height_IQR = daF.groupby(["SDMVSTRA", "SDMVPSU"])["BMXHT"].quantile(0.75) - da.groupby(["SDMVSTRA", "SDMVPSU"])["BMXHT"].quantile(0.25)
BMI_IQR = daF.groupby(["SDMVSTRA", "SDMVPSU"])["BMXBMI"].quantile(0.75) - da.groupby(["SDMVSTRA", "SDMVPSU"])["BMXBMI"].quantile(0.25)
print("age IQR: \n", age_IQR, "\n," "height IQR: \n ",  height_IQR, "\n", "BMI IQR: \n ", BMI_IQR)

print("Male case")
daM = data_pool = da.query('RIAGENDRx == "Male"').copy()
age_IQR = daM.groupby(["SDMVSTRA", "SDMVPSU"])["RIDAGEYR"].quantile(0.75) - da.groupby(["SDMVSTRA", "SDMVPSU"])["RIDAGEYR"].quantile(0.25)
height_IQR = daM.groupby(["SDMVSTRA", "SDMVPSU"])["BMXHT"].quantile(0.75) - da.groupby(["SDMVSTRA", "SDMVPSU"])["BMXHT"].quantile(0.25)
BMI_IQR = daM.groupby(["SDMVSTRA", "SDMVPSU"])["BMXBMI"].quantile(0.75) - da.groupby(["SDMVSTRA", "SDMVPSU"])["BMXBMI"].quantile(0.25)
print("age IQR: \n", age_IQR, "\n," "height IQR: \n ",  height_IQR, "\n", "BMI IQR: \n ", BMI_IQR)

print("Females case")
#daF = da.query('RIAGENDRx == "Female"').copy()
MVU1_values_list = sorted(daF['SDMVSTRA'].dropna().unique().tolist())  #creates a list with the unique values RIAGENDR assumes
MVU2_values_list =  sorted(daF['SDMVPSU'].dropna().unique().tolist())
#print(MVU2_values_list, MVU1_values_list)
three_items = ["RIDAGEYR", "BMXHT", "BMXBMI"]
for variable in three_items:
    lowest = float('inf') #infinty
    highest = float('-inf')
    #print("We now search the smallest and highest mean for", variable)
    for value1 in MVU1_values_list:
        for value2 in MVU2_values_list:
            data_pool = daF.query('SDMVSTRA == @value1 and SDMVPSU == @value2').copy()
            IQR = data_pool[variable].quantile(0.75) - data_pool[variable].quantile(0.25)
            if IQR < lowest:
                lowest = IQR
            if IQR > highest: #not elif because i may exit the loop for good, using the first if, if they are in order
                highest = IQR
    rounded_lowest = np.round(lowest, 3)
    rounded_highest = np.round(highest, 3)
    ratioHighLow = rounded_highest/rounded_lowest
    rounded_ratio = np.round(ratioHighLow, 3)
    print("The ratio between the highest and lowest IQR in", variable, "for Females is:", rounded_ratio)

print("Males case")
#daM = data_pool = da.query('RIAGENDRx == "Male"').copy()
MVU1_values_list = sorted(daM['SDMVSTRA'].dropna().unique().tolist())  #creates a list with the unique values RIAGENDR assumes
MVU2_values_list =  sorted(daM['SDMVPSU'].dropna().unique().tolist())
#print(MVU2_values_list, MVU1_values_list)
three_items = ["RIDAGEYR", "BMXHT", "BMXBMI"]
for variable in three_items:
    lowest = float('inf') #infinty
    highest = float('-inf')
    #print("We now search the smallest and highest mean for", variable)
    for value1 in MVU1_values_list:
        for value2 in MVU2_values_list:
            data_pool = daM.query('SDMVSTRA == @value1 and SDMVPSU == @value2').copy()
            IQR = data_pool[variable].quantile(0.75) - data_pool[variable].quantile(0.25)
            if IQR < lowest:
                lowest = IQR
            if IQR > highest: #not elif because i may exit the loop for good, using the first if, if they are in order
                highest = IQR
    rounded_lowest = np.round(lowest, 3)
    rounded_highest = np.round(highest, 3)
    ratioHighLow = rounded_highest / rounded_lowest
    rounded_ratio = np.round(ratioHighLow, 3)
    print("The ratio between the highest and lowest IQR in", variable, "for Males is:", rounded_ratio)
