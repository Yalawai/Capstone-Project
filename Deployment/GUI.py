import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import pandas as pd
import numpy
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler



class MedicalInsurancePricePredictionApp:
    def __init__(self, master):
        self.master = master
        self.master.title('Medical Insurance Price Prediction')
        self.data = pd.read_csv('Medical_insurance.csv')
        
          #we load the the trained model
        with open('Final_XGB_Model.pkl', 'rb') as fileReadStream:
            PredictionModel=pickle.load(fileReadStream)
            # Don't forget to close the filestream!
            fileReadStream.close()

        with open('DataForML.pkl', 'rb') as fileReadStream:
            DataForML=pickle.load(fileReadStream)
            # Don't forget to close the filestream!
            fileReadStream.close()

    
        self.DataForML = DataForML
        # Treating all the nominal variables at once using dummy variables
        DataForML_Numeric=pd.get_dummies(DataForML)

        # Adding Target Variable to the data
        DataForML_Numeric['charges']= self.data['charges']

        TargetVariable='charges'
        Predictors=['age', 'children', 'sex_female', 'sex_male', 'smoker_no', 'smoker_yes',
        'region_northeast', 'region_northwest', 'region_southeast',
        'region_southwest']

        X=DataForML_Numeric[Predictors].values
        y=DataForML_Numeric[TargetVariable].values

        # Choose between standardization and MinMAx normalization
        #PredictorScaler=StandardScaler()
        PredictorScaler=MinMaxScaler()

        # Storing the fit object for later reference
        self.PredictorScalerFit=PredictorScaler.fit(X)

        # Generating the standardized values of X
        X=self.PredictorScalerFit.transform(X)

        self.model = PredictionModel
        

        self.create_widgets()

    def create_widgets(self):

        label_age = tk.Label(self.master, text="age")
        label_age.pack()
        self.age = tk.Entry(self.master, bd=2, width="10")
        self.age.pack()

        label_children = tk.Label(self.master, text="children")
        label_children.pack()
        self.children = tk.Entry(self.master, bd=2, width="10")
        self.children.pack()

        label_gender = tk.Label(self.master, text="gender")
        label_gender.pack()
        options_gender = ["male", "female"]
        self.gender = tk.StringVar(self.master)
        self.gender.set("") # default value
        dropdown_menu = tk.OptionMenu(self.master, self.gender, *options_gender)
        dropdown_menu.pack()

        label_smoker = tk.Label(self.master, text="smoker")
        label_smoker.pack()
        self.smoker = tk.Entry(self.master, bd=2, width="10")
        self.smoker.pack()

        label_region = tk.Label(self.master, text="region")
        label_region.pack()
        options_region = ["northeast", "northwest","southeast", 'southwest']
        self.region = tk.StringVar(self.master)
        self.region.set("") # default value
        dropdown_menu1 = tk.OptionMenu(self.master, self.region, *options_region)
        dropdown_menu1.pack()

        predict_button = tk.Button(self.master, text='predict', command=self.predict_price)
        predict_button.pack()


        

    def predict_price(self):
        InputData=pd.DataFrame(data=[[float(self.age.get()), float(self.children.get()), str(self.gender.get()), bool(self.smoker.get()), str(self.region.get())]],columns=['age','children','sex','smoker', 'region'])
    
        Num_Inputs=InputData.shape[0]

        # Making sure the input data has same columns as it was used for training the model
        # Also, if standardization/normalization was done, then same must be done for new input

        # Appending the new data with the Training data
        DataForML=pd.read_pickle('DataForML.pkl')
        #InputData=InputData.append(DataForML, ignore_index=True)
        InputData = pd.concat([InputData, DataForML], ignore_index=True)

        # Generating dummy variables for rest of the nominal variables
        InputData=pd.get_dummies(InputData)

        # Maintaining the same order of columns as it was during the model training
        Predictors=['age', 'children', 'sex_female', 'sex_male', 'smoker_no', 'smoker_yes',
        'region_northeast', 'region_northwest', 'region_southeast',
        'region_southwest']

        # Generating the input values to the model
        X=InputData[Predictors].values[0:Num_Inputs]

        # Generating the standardized values of X since it was done while model training also
        X = self.PredictorScalerFit.transform(X)

        Prediction= self.model.predict(X)
        PredictionResult=pd.DataFrame(Prediction, columns=['Prediction'])
        messagebox.showinfo('Predicted Price', f'The Medical Insurance price is {PredictionResult}')

if __name__ == '__main__':
    root = tk.Tk()
    app = MedicalInsurancePricePredictionApp(root)
    root.mainloop()
