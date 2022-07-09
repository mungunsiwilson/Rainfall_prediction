#import everything from tkinter

#import tkinter as tk
from base64 import encode
from copyreg import pickle
import os
from tkinter import Canvas, filedialog, Text
from tkinter import *
from matplotlib.pyplot import text
from PIL import ImageTk, Image
from numpy import place
import pickle
import pandas as pd
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

list_num_col = []
list_cat_col = []

def finalpage():
    page3 = Tk()
    page3.title("Weather Forecast")
    F = Canvas(page3, height=400, width=600)
    F.pack()
    #print(list_cat_col) 
    list1 = [list_num_col]
    lv=['Location','MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday']
    df = pd.DataFrame(list1, index=[0], columns=lv)

    #scale numerical data
    Test1 = pd.read_csv("file_name2.csv")
    num_cols = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm']
    scaler = MinMaxScaler()
    scaler.fit(Test1)
    df[num_cols] = scaler.transform(df[num_cols])
    #encode non numeric data
    Test = pd.read_csv('file_name.csv')
    '''le = LabelEncoder()
    cat_cols = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
    for a in cat_cols:
        df[a] = le.fit_transform(df[a])
    '''
    cat_cols = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoder.fit(Test)
    #generate colmn names for our new encoded cols
    encoded_cols = list(encoder.get_feature_names_out(cat_cols))

    df[encoded_cols] = encoder.transform(df[cat_cols])
    df=df.drop(['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday'], axis=1)

    with open("model_1", 'rb') as f:
        mp = pickle.load(f) 
        
    #print(mp.predict(df))
    lb23 = Label(page3, text="The probability that its going to rain tommorow is:", font=("Italics", 18))
    lb23.place(x=55, y=10)
    T = Text(page3, height=5, width=30, font=("Italics", 12))
    T.place(x=150, y=100)
    T.insert(END, mp.predict(df))
    T.insert(END, "\n")
    T.insert(END, "\n")
    T.insert(END, mp.predict_proba(df))
    B1 = Button(page3, text="Exit", padx=10, pady=2, fg="white", bg="grey",activebackground="blue", font=("Italics", 11), command= page3.destroy)
    B1.place(x=260,y=360)

def Nextpage():
    def getinput():
        
        a = param1.get(1.0,"end-1c")
        list_num_col.append(a)
                
        b = float(param4.get(1.0,"end-1c"))
        list_num_col.append(b)
                      
        c = float(param5.get(1.0,"end-1c"))
        list_num_col.append(c)
                        
        d = float(param6.get(1.0,"end-1c"))
        list_num_col.append(d)
                        
        e = float(param7.get(1.0,"end-1c"))
        list_num_col.append(e)
                        
        f = float(param8.get(1.0,"end-1c"))
        list_num_col.append(f)
                        
        g = param9.get(1.0,"end-1c")
        list_num_col.append(g)
                        
        h = float(param2.get(1.0,"end-1c"))
        list_num_col.append(h)
                        
        i = param10.get(1.0,"end-1c")
        list_num_col.append(i)
                        
        j = param11.get(1.0,"end-1c")
        list_num_col.append(j)
                        
        k = float(param12.get(1.0,"end-1c"))
        list_num_col.append(k)
                        
        l = float(param13.get(1.0,"end-1c"))
        list_num_col.append(l)
                        
        m = float(param14.get(1.0,"end-1c"))
        list_num_col.append(m)
                        
        n = float(param15.get(1.0,"end-1c"))
        list_num_col.append(n)
                        
        o = float(param3.get(1.0,"end-1c"))
        list_num_col.append(o)
                        
        p = float(param16.get(1.0,"end-1c"))
        list_num_col.append(p)
                        
        q = float(param17.get(1.0,"end-1c"))
        list_num_col.append(q)
                        
        r  = float(param18.get(1.0,"end-1c"))
        list_num_col.append(r)
                        
        s = float(param19.get(1.0,"end-1c"))
        list_num_col.append(s)
                        
        t = float(param20.get(1.0,"end-1c"))
        list_num_col.append(t)
                        
        u = param21.get(1.0,"end-1c")
        list_num_col.append(u)
    page2 = Tk()
    page2.title("Weather Forecast")
     
    #image =Image.open('C:\\Users\\PC\\Desktop\\Machine learning project\\image.jpg')
    #bg1 = ImageTk.PhotoImage(image, master=page2)
    
    #Canvas = Canvas(page2, height=400, width=600)
    #from tkinter import Canvas
    C1 = Canvas(page2, height=400, width=600)
    C1.pack()
    #lb = Label(page2, image=bg1)
    #lb.place(x=0,y=0)
    lb2 = Label(page2, text="Fill the boxes below, save and click next", font=("Italics", 18))
    lb2.place(x=55, y=10)
    #col1
    param1 = Text(page2, width=12, height=0.5)
    param1.place(x=80, y=50)
    l_p1 = Label(page2, text="Location", font=("Italics", 11))
    l_p1.place(x=10, y=50)
 
    
    param4 = Text(page2, width=12, height=0.5)
    param4.place(x=80, y=90)
    l_p4 = Label(page2, text="MinTemp", font=("Italics", 11))
    l_p4.place(x=10, y=90)

        
    param5 = Text(page2, width=12, height=0.5)
    param5.place(x=80, y=130)
    l_p5 = Label(page2, text="MaxTemp", font=("Italics", 11))
    l_p5.place(x=10, y=130)
        
    param6 = Text(page2, width=12, height=0.5)
    param6.place(x=80, y=170)
    l_p6 = Label(page2, text="Rainfall", font=("Italics", 11))
    l_p6.place(x=10, y=170)
        
    param7 = Text(page2, width=12, height=0.5)
    param7.place(x=80, y=210)
    l_p7 = Label(page2, text="Evaporat", font=("Italics", 11))
    l_p7.place(x=10, y=210)
        
    param8 = Text(page2, width=12, height=0.5)
    param8.place(x=80, y=250)
    l_p8 = Label(page2, text="Sunshine", font=("Italics", 11))
    l_p8.place(x=5, y=250)
        
    param9 = Text(page2, width=12, height=0.5)
    param9.place(x=80, y=290)
    l_p9 = Label(page2, text="WinGd9am", font=("Italics", 11))
    l_p9.place(x=0, y=290)
    
    #col2
    param2 = Text(page2, width=12, height=0.5)
    param2.place(x=270, y=50)
    l_p2 = Label(page2, text="WindGspd", font=("Italics", 11))
    l_p2.place(x=190, y=50)
    
    param10 = Text(page2, width=12, height=0.5)
    param10.place(x=270, y=90)
    l_p10 = Label(page2, text="Windir9am", font=("Italics", 11))
    l_p10.place(x=190, y=90)
        
    param11 = Text(page2, width=12, height=0.5)
    param11.place(x=270, y=130)
    l_p11 = Label(page2, text="Windir3pm", font=("Italics", 11))
    l_p11.place(x=185, y=130)
        
    param12 = Text(page2, width=12, height=0.5)
    param12.place(x=270, y=170)
    l_p12 = Label(page2, text="Windsp9am", font=("Italics", 11))
    l_p12.place(x=185, y=170)
        
    param13 = Text(page2, width=12, height=0.5)
    param13.place(x=270, y=210)
    l_p13 = Label(page2, text="Windsp3pm", font=("Italics", 11))
    l_p13.place(x=185, y=210)
        
    param14 = Text(page2, width=12, height=0.5)
    param14.place(x=270, y=250)
    l_p14 = Label(page2, text="Humid9am", font=("Italics", 11))
    l_p14.place(x=190, y=250)
        
    param15 = Text(page2, width=12, height=0.5)
    param15.place(x=270, y=290)
    l_p15 = Label(page2, text="Humid3pm", font=("Italics", 11))
    l_p15.place(x=190, y=290)
    #col3
    param3 = Text(page2, width=12, height=0.5)
    param3.place(x=450, y=50)
    l_p3 = Label(page2, text="Presur9am", font=("Italics", 11))
    l_p3.place(x=370, y=50)
        
    param16 = Text(page2, width=12, height=0.5)
    param16.place(x=450, y=90)
    l_p16 = Label(page2, text="Presur3pm", font=("Italics", 11))
    l_p16.place(x=370, y=90)
        
    param17 = Text(page2, width=12, height=0.5)
    param17.place(x=450, y=130)
    l_p17 = Label(page2, text="Cloud9am", font=("Italics", 11))
    l_p17.place(x=375, y=130)
        
    param18 = Text(page2, width=12, height=0.5)
    param18.place(x=450, y=170)
    l_p18 = Label(page2, text="Cloud3pm", font=("Italics", 11))
    l_p18.place(x=375, y=170)
        
    param19 = Text(page2, width=12, height=0.5)
    param19.place(x=450, y=210)
    l_p19 = Label(page2, text="Temp9am", font=("Italics", 11))
    l_p19.place(x=375, y=210)
        
    param20 = Text(page2, width=12, height=0.5)
    param20.place(x=450, y=250)
    l_p20 = Label(page2, text="Temp3pm", font=("Italics", 11))
    l_p20.place(x=375, y=250)
        
    param21 = Text(page2, width=12, height=0.5)
    param21.place(x=450, y=290)
    l_p21 = Label(page2, text="Raintody", font=("Italics", 11))
    l_p21.place(x=375, y=290)
    #predict button
    B = Button(page2, text="Next", padx=10, pady=2, fg="white", bg="grey",activebackground="blue", font=("Italics", 11), command=finalpage)
    B.place(x=260,y=360)
    save = Button(page2, text="save", padx=10, pady=2, fg="white", bg="grey",activebackground="blue", font=("Italics", 11), command= getinput)
    save.place(x=260,y=320)


root = Tk()
root.title("Weather Forecast")
img =Image.open('C:\\Users\\PC\\Desktop\\code\\image1.jpg')
bg = ImageTk.PhotoImage(img)

#define the size of the window
C = Canvas(root, height=400, width=600, bg="black")
C.pack()
#creating a frame

#frame = Frame(root)
#frame.place(relwidth=0.8, relheight=0.9, relx=0.1, rely=0.1)
label = Label(root, image=bg)
label.place(x=0,y=0)
l1 = Label(root, text="Hello! welcome to Wilson's weather forcast", font=("Italics", 18))
#l1.pack()
l1.place(x=55, y=10)
l2 = Label(root, text="To predict if its going to rain", font=("Italics", 12))
l2.place(x=170, y=100)

l5 = Label(root, text="tomorrow, click the link below to get todays'", font=("Italics", 12))
l5.place(x=140, y=125)

l6 = Label(root, text="weather informations", font=("Italics", 12))
l6.place(x=200, y=150)

l3 = Label(root, text="https://www.eldersweather.com.au/", font=("Italics", 12), fg="blue")
l3.place(x=170, y=200)

l4 = Label(root, text="click on next and enter the informations obtained from the link above", font=("Italics", 12), fg="black")
l4.place(x=50, y=250)
#creating a button
Button1 = Button(root, text="Next", padx=10, pady=2, fg="white", bg="grey",activebackground="blue", font=("Italics", 11), command=Nextpage)
#Button.pack()
Button1.place(x=260,y=340)

root.mainloop()
