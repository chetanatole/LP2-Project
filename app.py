from Project import AirQuality
import matplotlib.pyplot as plt
import itertools
import numpy as np
from PIL import ImageTk, Image
import threading
import time
from tkinter import *
from tkinter.ttk import Progressbar
from tkinter.filedialog import askopenfilename
obj = AirQuality()
window = Tk()   
window.title("Air Quality Predictor")
window.state('zoomed')
window.configure(bg="White")
frame1 = Frame(window,bg="White")
frame2 = Frame(window,bg="White")
frame1.pack()
classlabels=['Good','Moderate','Poor','Satisfactory','Severe','Very Poor']
def plot_confusion_matrix(cm,title, classes=classlabels,
                          cmap=plt.cm.Blues):

    plt.figure(figsize=(5,4.8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

#     print(cm)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(title+'.jpg')

def getCSV():
    csvfilename = askopenfilename(title = "Select CSV File")
    if(csvfilename != ""):
        Label(frame1, text = 'File Selected Successfully ',bg="White",font =('Verdana', 15), fg="red").pack( pady=20 )        
        Button(frame1, text="Train",command = startTrainThread,width=20).pack(pady=40)
#         print(csvfilename)
        obj.readCsv(csvfilename)


    
def startTrainThread():
    progress.pack(pady=10)
    th=threading.Thread(target=train)
    th.start()

def train():
    obj.trainDT()
    progress['value'] = 25
    window.update_idletasks()
    obj.trainXGB()
    progress['value'] = 50
    window.update_idletasks()
    obj.trainRF()
    progress['value'] = 75
    window.update_idletasks()
    obj.trainSVM()
    progress['value'] = 100
    window.update_idletasks()
    time.sleep(1)
    frame1.destroy()
    frame2.pack()    
    
def test():
    predictFrame1.pack_forget()
    predictFrame2.pack_forget()
    predictFrame3.pack_forget()
    predictFrame4.pack_forget()
    predictFrame5.pack_forget()
    predictResultFrame.pack_forget()
    resultFrame.pack_forget()
    testFrame.pack()
    testPanel.pack()

def testClassifier():
    selector = var.get()
    if selector=='Random Forest':
        cm,a,precision,recall,f1=obj.RandomForest()
    elif selector=='Decision Tree':
        cm,a,precision,recall,f1=obj.DecisionTree()
    elif selector=='Support Vector Machine':
        cm,a,precision,recall,f1=obj.SVC()
    else:
        cm,a,precision,recall,f1=obj.XGB()
    acc_label.config(text="Accuracy :" +str(a))
    precision_label.config(text="Precision : "+str(precision))
    recall_label.config(text="Recall : "+str(recall))
    f1score_label.config(text="F1 Score : "+str(f1))
    plot_confusion_matrix(cm,title=selector)
    from PIL import ImageTk, Image
    im = Image.open(selector+'.jpg')
#     newsize=(432,350)
#     im = im.resize(newsize)
    img = ImageTk.PhotoImage(im)
    cm_label.config(image=img)
    cm_label.image=img
    resultFrame.pack()

#     print(cm,a,precision,recall,f1)
        
        

        
def predict():
    testFrame.pack_forget()
    testPanel.pack_forget()
    resultFrame.pack_forget()
    predictResultFrame.pack_forget()
    predictFrame1.pack()
    predictFrame2.pack()
    predictFrame3.pack()
    predictFrame4.pack()
    predictFrame5.pack()
    
def predictResult():
    result = obj.predict(City_var.get(),Date_var.get(),PM25_var.get(),PM10_var.get(),NO_var.get(),
                        NO2_var.get(),NOx_var.get(),NH3_var.get(),CO_var.get(),SO2_var.get(),
                        O3_var.get(),Benzene_var.get(),Toluene_var.get(),Xylene_var.get(),AQI_var.get())
    print(result)
    decision_tree_label.config(text="Decision Tree : "+str(result[2]))
    random_forest_label.config(text="Random Forest : "+str(result[0]))
    SVM_label.config(text="Support Vector Machine : "+str(result[1]))
    XGB_label.config(text="eXtreme Gradient Boosting : "+str(result[3]))
    predictResultFrame.pack()




#Frame 1
Label(frame1, text = 'Select CSV ',bg="White",font =('Verdana', 15)).pack( pady=20 )
Button(frame1, text="Select CSV file", command=getCSV,width=20).pack(pady=10)
progress = Progressbar(frame1, orient = HORIZONTAL, length = 400, mode = 'determinate')


#Frame 2
Button(frame2, text="Test",command=test,width=20 ).pack(pady=40,side=LEFT,padx=175)
Button(frame2, text="Predict",command=predict,width=20).pack(pady=40,side=RIGHT,padx=175)


#TestFrame
testFrame=Frame(window,bg="White")
var = StringVar()
var.set('Random Forest')
choices = { 'Random Forest','Decision Tree','Support Vector Machine','eXtreme Gradient Boosting'}
popupMenu = OptionMenu(testFrame, var, *choices)
Label(testFrame, text = 'Testing ',bg="White",font =('Verdana', 15)).pack(pady=20)
Label(testFrame, text = 'Select Classifier',bg="White",font =('Verdana', 15)).pack( side=LEFT,padx=20,pady=10 )
popupMenu.pack(side=LEFT)

#TestPanel
testPanel=Frame(window,bg="White")
Button(testPanel, text="Test",command=testClassifier,width=20).pack(pady=20)

#TestResultFrame
resultFrame=Frame(window,bg='white',pady=10)
acc_label = Label(resultFrame, text = "",bg="White",font =('Verdana', 10))
acc_label.pack( pady=10 )
precision_label = Label(resultFrame, text = "",bg="White",font =('Verdana', 10))
precision_label.pack( pady=10 )
recall_label = Label(resultFrame, text = "",bg="White",font =('Verdana', 10))
recall_label.pack( pady=10 )
f1score_label = Label(resultFrame, text = "",bg="White",font =('Verdana', 10))
f1score_label.pack( pady=10 )
from PIL import ImageTk, Image
# img = ImageTk.PhotoImage(Image.open("mkbhd1.jpg"))
cm_label = Label(resultFrame)
# cm_label.image=img
cm_label.pack(pady=10)

#predictFrame
predictFrame1=Frame(window,bg="White",pady=10)
predictFrame2=Frame(window,bg="White",pady=10)
predictFrame3=Frame(window,bg="White",pady=10)
predictFrame4=Frame(window,bg="White",pady=10)
predictFrame5=Frame(window,bg="White",pady=10)

City_var = StringVar()
Date_var = StringVar()
PM25_var = StringVar()
PM10_var = StringVar()
NO_var = StringVar()
NO2_var = StringVar()
NOx_var = StringVar()
NH3_var = StringVar()
CO_var = StringVar()
SO2_var = StringVar()
O3_var = StringVar()
Benzene_var = StringVar()
Toluene_var = StringVar()
Xylene_var = StringVar()
AQI_var = StringVar()

Label(predictFrame1, text = 'City',bg="White").pack( pady=10,side=LEFT,padx=10)
City=Entry(predictFrame1,textvariable=City_var).pack(pady=10,side=LEFT,padx=15)
Label(predictFrame1, text = 'Date',bg="White").pack( pady=10,side=LEFT,padx=10)
Date=Entry(predictFrame1,textvariable=Date_var).pack(pady=10,side=LEFT,padx=15)
Label(predictFrame1, text = 'PM 2.5',bg="White").pack( pady=10,side=LEFT,padx=10)
PM25=Entry(predictFrame1,textvariable=PM25_var).pack(pady=10,side=LEFT,padx=15)
Label(predictFrame1, text = 'PM 10',bg="White").pack( pady=10,side=LEFT,padx=10)
PM10=Entry(predictFrame1,textvariable=PM10_var).pack(pady=10,side=LEFT,padx=15,fill=BOTH)
Label(predictFrame2, text = 'NO',bg="White").pack( pady=10,side=LEFT,padx=10)
NO=Entry(predictFrame2,textvariable=NO_var).pack(pady=10,side=LEFT,padx=15)
Label(predictFrame2, text = 'NO2',bg="White").pack( pady=10,side=LEFT,padx=10)
NO2=Entry(predictFrame2,textvariable=NO2_var).pack(pady=10,side=LEFT,padx=15)
Label(predictFrame2, text = 'NOx',bg="White").pack( pady=10,side=LEFT,padx=10)
NOx=Entry(predictFrame2,textvariable=NOx_var).pack(pady=10,side=LEFT,padx=15)
Label(predictFrame2, text = 'NH3',bg="White").pack( pady=10,side=LEFT,padx=10)
NH3=Entry(predictFrame2,textvariable=NH3_var).pack(pady=10,side=LEFT,padx=15)
Label(predictFrame3, text = 'CO',bg="White").pack( pady=10,side=LEFT,padx=10)
CO=Entry(predictFrame3,textvariable=CO_var).pack(pady=10,side=LEFT,padx=15)
Label(predictFrame3, text = 'SO2',bg="White").pack( pady=10,side=LEFT,padx=10)
SO2=Entry(predictFrame3,textvariable=SO2_var).pack(pady=10,side=LEFT,padx=15)
Label(predictFrame3, text = 'O3',bg="White").pack( pady=10,side=LEFT,padx=10)
O3=Entry(predictFrame3,textvariable=O3_var).pack(pady=10,side=LEFT,padx=15)
Label(predictFrame3, text = 'Benzene',bg="White").pack( pady=10,side=LEFT,padx=10)
Benzene=Entry(predictFrame3,textvariable=Benzene_var).pack(pady=10,side=LEFT,padx=15)
Label(predictFrame4, text = 'Toluene',bg="White").pack( pady=10,side=LEFT,padx=10)
Toluene=Entry(predictFrame4,textvariable=Toluene_var).pack(pady=10,side=LEFT,padx=15)
Label(predictFrame4, text = 'Xylene',bg="White").pack( pady=10,side=LEFT,padx=10)
Xylene=Entry(predictFrame4,textvariable=Xylene_var).pack(pady=10,side=LEFT,padx=15)
Label(predictFrame4, text = 'AQI',bg="White").pack( pady=10,side=LEFT,padx=10)
AQI=Entry(predictFrame4,textvariable=AQI_var).pack(pady=10,side=LEFT,padx=15)
Button(predictFrame5, text="Predict",command=predictResult,width=20).pack(pady=20)

#PredictResultFrame
predictResultFrame=Frame(window,bg='white',pady=10)
decision_tree_label = Label(predictResultFrame, text = "",bg="White",font =('Verdana', 10))
decision_tree_label.pack( pady=10 )
random_forest_label = Label(predictResultFrame, text = "",bg="White",font =('Verdana', 10))
random_forest_label.pack( pady=10 )
SVM_label = Label(predictResultFrame, text = "",bg="White",font =('Verdana', 10))
SVM_label.pack( pady=10 )
XGB_label = Label(predictResultFrame, text = "",bg="White",font =('Verdana', 10))
XGB_label.pack( pady=10 )
window.mainloop()  