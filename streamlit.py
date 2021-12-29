# Import library
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Rectangle
import time
import calendar
from pygame import mixer

import tensorflow as tf
from tensorflow import keras

warning = 'sound/alarm.mp3'
alarm = 'sound/alarm_2.mp3'

# Import Model and prediction classes
from tensorflow import keras
model = keras.models.load_model('article_model/article_vgg16.h5')


def load_image(uploaded_file):
    image = uploaded_file.read()
    image = tf.convert_to_tensor(image)
    image = tf.image.decode_jpeg(image,3)               
    image = tf.image.resize(image,[224,224])               
    image = tf.cast(image,tf.float32)  
    image = image / 255
    image = tf.expand_dims(image, axis = 0)
    return image

class_dict = {  1: 'crescent_gap',
                2: 'crease',
                3: 'silk_spot',
                4: 'water_spot',
                5: 'welding_line',
                6: 'inclusion',
                7: 'oil_spot',
                8: 'waist_folding',
                9: 'rolled_pit',
                10: 'punching_hole'
             }

# Streamlit UI

st.set_page_config(
     page_title="CS_Final project: Metal defect detection",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",
 )



#  Side bar menu
menu = ['About me','Existing Situation', 'Model', 'Demo', 'Future work']
choice = st.sidebar.radio('Navigator', menu)

if choice == 'References':
    st.title('References')

elif choice == 'About me':
    st.title('A little bit about me')
    bio_3, bio_1, bio_2 = st.columns([0.5, 2,3])
    with bio_1:
        st.image('photo\photo.PNG',width = 200)
    with bio_2:
        st.markdown(' ')
        st.markdown(' ')
        st.markdown('Good day everybody, my name is **Cuong Vo**. In the past I was an **Industrial Systems engineer** with the passion to **optimizing** all aspects of the manufacturing plant especially in **production cost reduction**')    
        st.markdown('I had a chance to work with two major companies in their own fields, therefore I have seen the potentials as well as the limitations of applying **machine learning** in factories.')
    st.subheader('Places that I have worked with')
    col1, col2 = st.columns(2)    
    with col1:
        st.image('photo/colgate.jpg')
        st.markdown('While I worked with **<span style="color:Blue;">Colgate Palmolive</span>** which manufactured toothbrushs in Vietnam. The production lines included many machines with **<span style="color : red;">a lot of human interference</span>** such as injection molding, tufting and enrounding, and packagaging machines.',unsafe_allow_html=True)
        st.markdown('Quality controls of the product was done by Quality controllers in a set period of time or workers while they were working. Of course when there were humans involved, mistakes will be made and the defects cost were always our main target to strike down but it was so difficult.')

    with col2:
        st.image('photo/Heineken.jpg')
        st.markdown('**<span style="color : rgb(0, 105, 0);">Heineken</span>** was a different story, all machines here were so highly automated with **<span style="color : red;">little human interactions</span>**. Quality control were done by machines as well as human in some importants stages.',unsafe_allow_html=True)
        st.markdown('However, since there are many inspection machines providers for example Empty bottle inspection stages already have at least 3 kinds of providers Heufts, Krones, or KHS. And while aiming to integrate all the production line into 1 whole systems for production control, those variation of machines pose a challenge.')
    st.markdown('Therefore, I have viewpoints from two sides of a manufacturing plant could have **Human** versus **Robot**')

elif choice == 'Existing Situation':
    st.title('1. The problem')
    st.markdown('When it comes down to the production cost reduction problems, there are a variety of methods that we can apply. On the one hand, we can use reduce the actual production cost such as material cost, labour cost, or utility cost. On the other hand, we can try to minimize the wasted cost such as reworks, defects, or scraps cost. This project will be focused on applying machine learning tools and methodology to reduce the latter suggestions.')
    st.markdown('Quality control of a product is either monitored by human during production or using real-time detection from the machines. However, since those checks are not 100% perfect every single time for example fatigues of the workers or defects of the inspection machines. And when the defects are not detected in time, which leads to a series of scraps or unuseable products and that costs money to the coroperation.')
    st.markdown('Currently, some solutions may be implemented in the shopfloor such as crosschecking quality of product by Quality Assurance personels and workers, maintenance schedule for production machines, or using costly inspection machine with high state of reliability (with maintenance schedule). However, this may lead to a type of waste according the Lean concept which is Overprocessing where more works or higher quality than is required by the customers.')
    st.markdown('Therefore, I try another approach in order to solve the problem.')

    st.title('2. Proposed solution')
    st.markdown('In order to reduce the time and money while keeping the production flows smoothly. The following solutions and reasons for it will be discussed below.')
    st.markdown('By using a camera to capture the picture of the final products in the controlled environment. We can have the image of the product with high definition details. By applying machine learning to learn where are the defects and what type of that defect, we can have several benefits from it.')
    st.markdown('<ol> <li>Fast and reliable defect detection.</li> <li>Keep track of the defect type in history log.</li> <li>Notice the maintenance personels if there are any problem with the machines.</li> <li>Listing the tools needed for the repair.</li> </ol>', unsafe_allow_html=True)

elif choice == 'Model':
    st.title('1. GC10-DET')
    st.markdown('The project is based on the dataset **GC10-DET** uploaded by Via.')
    st.markdown('[https://www.kaggle.com/zhangyunsheng/defects-class-and-location](https://www.kaggle.com/zhangyunsheng/defects-class-and-location)')
    st.markdown('**GC10-DET** is the surface defect dataset collected in a real industry. It contains ten types of surface defects, i.e:')
    st.markdown('<ul>  <li>Punching (Pu): 219 images.</li>  <li>Weld line (Wl): 273 images.</li>  <li>Crescent gap (Cg): 226 images.</li>  <li>Water spot (Ws): 289 images.</li>  <li>Oil spot (Os): 204 images.</li>  <li>Silk spot (Ss): 650 images.</li>  <li>Inclusion (In): 216 images.</li>  <li>Rolled pit (Rp): 31 images.</li>  <li>Crease (Cr): 52 images.</li>  <li>Waist folding (Wf): 146 images.</li> </ul>', unsafe_allow_html=True)
    st.markdown('The collected defects are on the surface of the steel sheet. The dataset includes **2280 gray-scale images** and its label for type of defect and the true bounding box.')
    sample_1, sample_2 = st.columns(2)
    with sample_1:
        st.image('photo/sample_1.PNG')
    with sample_2:
        st.image('photo/sample_2.png')

    st.title('2. Datapipeline')
    st.header('2.1. Preprocessing data: Train, Validation, Test split')
    st.markdown('In order to keep the integrity of the prediction results. I have decided to seperate the dataset into train, validation, and test set.')
    st.markdown('<ul>  <li>Train set (77.5%): 1767 images.</li>  <li>Validation set (17.5%): 399 images.</li>  <li>Test set (5%): 114 images.</li> </ul>', unsafe_allow_html = True)
    st.markdown('All the set are shuffled and divided into a batch size of 32.')
    st.header('2.2. Model structure')
    st.markdown('The aim of the model is trying to use Convolutional Neural Network (CNN) such as VGG16, Xception, InceptionResNetV2, ResNet152V2 and choosing the best performance models for prediction.')
    st.markdown('There will be 2 parallel branches for the model to learn:')
    st.markdown('<ul> <li> Reduce the MSE of the coordinations of 4 points of the bounding box. </li> <li> Reduce the loss of the labels using categorical crossentropy. </li> </ul>', unsafe_allow_html= True)
    st.markdown('**MAE** and **Accuracy** are the metrics used to evaluate the performance of the models. By doing this, the two branches compliment to each others resulting higher accuracy predictions.')

    st.markdown('VGG16 architecture is used to demonstrate this solution:')
    adjust_1, adjust_2, adjust_3 = st.columns([0.7,5,0.2])
    with adjust_2:
        st.image('photo/architecture.png')

    st.header('2.3. Result and choosing models')
    st.markdown('VGG16 performance metrics are plotted as the example:')
    chart_1, chart_2, chart_3 = st.columns(3)
    with chart_1:
        st.image('photo/chart_1.PNG')
        st.markdown("<p style = 'text-align: center;'>1. Total MSE (bounding box)</p>", unsafe_allow_html=True)
    with chart_2:
        st.image('photo/chart_2.PNG')
        st.markdown("<p style = 'text-align: center;'>2. Validation MAE</p>", unsafe_allow_html=True)
    with chart_3:
        st.image('photo/chart_3.PNG')
        st.markdown("<p style = 'text-align: center;'>3. Validation class ACC</p>", unsafe_allow_html=True)

    st.markdown('After all the models are trained with 200 epochs each, the following table depicts the results:')
    st.image('photo/result.PNG')

    st.markdown('Since all the models can predict the defect classes with high accuracy (1.00). I choose the model with the lowest size for **prediction speed**.')
    st.markdown('<ul> <li>VGG16.h5: 197 Mb </li> <li>Xception.h5: 306 Mb </li> <li>Inception.h5: 688 Mb </li> <li>ResNet.h5: 746 Mb </li> </ul>', unsafe_allow_html=True)
    st.markdown('**VGG16 model** has the smallest size, therefore I will choose this model to predict the future images.')

elif choice == 'Demo':
    st.title('Predict Defect')


    photo_uploaded = st.file_uploader('Upload photo here', ['png', 'jpeg', 'jpg'])
    if photo_uploaded!=None:
        
        image_predict = load_image(photo_uploaded)    
        out1, out2, out3, out4, label = model.predict(image_predict) 

        # Prediction
        plt.figure(figsize = (10, 24))
        plt.imshow(keras.preprocessing.image.array_to_img(image_predict[0]))    
        pred_imglist = []
        pred_imglist.append(class_dict[np.argmax(label) + 1])
        plt.title(pred_imglist)
        xmin, ymin, xmax, ymax = out1*224, out2*224, out3*224, out4*224
        rect = Rectangle((xmin,ymin), (xmax - xmin), (ymax - ymin), fill = False, color = "r") 
        
        ax = plt.gca()
        ax.axes.add_patch(rect)   
        pred_imglist = []
        pred_imglist.append(class_dict[np.argmax(label)+1])
        plt.title(pred_imglist)
        st.set_option('deprecation.showPyplotGlobalUse', False)
     

        dashboard_1, dashboard_2 = st.columns(2)
        
        
        # Get timestamp and defect for log table

        ts = time.time()
        ts_key = round(ts)

        log_entry = class_dict[np.argmax(label)+1]

        if ts_key not in st.session_state:
            st.session_state[ts_key] = log_entry       

        body_log_sub = str()
        with dashboard_1:
            st.pyplot(plt.show())
        with dashboard_2:
            if st.button('Clear history'):
                for key in st.session_state.keys():
                    del st.session_state[key]
            for key in sorted(st.session_state.keys(), reverse=True):
                ts = time.ctime(int(key))
                body_log_sub += """ <tr>
                                        <td style="text-align:center">""" + str(ts) + """</td>
                                        <td style="text-align:center">""" + st.session_state[key] + """</td>
                                    </tr>
                                """
            body_log = """
                            <table style = "margin-left:auto;margin-right:auto;"> 
                            <tr>
                                <td> <b>Time stamp</b> </td>
                                <td> <b>Defect</b> </td> 
                            </tr>
                        """ + body_log_sub + """</table>"""


            st.markdown(body_log,unsafe_allow_html=True)
        text = log_entry + ' has been found.'
        st.header('Machine status:')
        
        # Display Crease DEFECT
        if log_entry == 'crease':
            st.subheader(text)
            space_1,space_2,space_3 = st.columns([2,5,0.3])
            with space_2:
                st.image('photo\machine\machine_1.png')
            mixer.init()
            mixer.music.load(warning)
            mixer.music.play()
            st.markdown('The part 1 is moderately damaged, please prepare the following tools to repair:')
            col_1, col_2, col_3 = st.columns(3)
            with col_1:
                st.markdown("<p style = 'text-align: center;'>Tool 1.1</p>", unsafe_allow_html=True)
                st.image('photo\photo_tool/1_1.jpg')
            with col_2:
                st.markdown("<p style = 'text-align: center;'>Tool 1.2</p>", unsafe_allow_html=True)
                st.image('photo\photo_tool/1_2.jpg')      
            with col_3:
                st.markdown("<p style = 'text-align: center;'>Tool 1.3</p>", unsafe_allow_html=True)
                st.image('photo\photo_tool/1_3.jpg')        
        # Display Cresent_gap defect
        elif log_entry == 'crescent_gap':
            st.subheader(text)
            space_1,space_2,space_3 = st.columns([2,5,0.3])
            with space_2:
                st.image('photo\machine\machine_2.png')
            mixer.init()
            mixer.music.load(warning)
            mixer.music.play()
            st.markdown('The part 2 is moderately damaged, please prepare the following tools to repair:')
            col_1, col_2, col_3 = st.columns(3)
            with col_1:
                st.markdown("<p style = 'text-align: center;'>Tool 2.1</p>", unsafe_allow_html=True)
                st.image('photo\photo_tool/2_1.jpg')
            with col_2:
                st.markdown("<p style = 'text-align: center;'>Tool 2.2</p>", unsafe_allow_html=True)
                st.image('photo\photo_tool/2_2.jpg')      
            with col_3:
                st.markdown("<p style = 'text-align: center;'>Tool 2.3</p>", unsafe_allow_html=True)
                st.image('photo\photo_tool/2_3.jpg')
        # Display Inclusion
        elif log_entry == 'inclusion':
            st.subheader(text)
            space_1,space_2,space_3 = st.columns([2,5,0.3])
            with space_2:
                st.image('photo\machine\machine_3.png')
            mixer.init()
            mixer.music.load(warning)
            mixer.music.play()
            st.markdown('The part 3 is moderately damaged, please prepare the following tools to repair:')
            col_1, col_2, col_3 = st.columns(3)
            with col_1:
                st.markdown("<p style = 'text-align: center;'>Tool 3.1</p>", unsafe_allow_html=True)
                st.image('photo\photo_tool/3_1.jfif')
            with col_2:
                st.markdown("<p style = 'text-align: center;'>Tool 3.2</p>", unsafe_allow_html=True)
                st.image('photo\photo_tool/3_2.jfif')      
            with col_3:
                st.markdown("<p style = 'text-align: center;'>Tool 3.3</p>", unsafe_allow_html=True)
                st.image('photo\photo_tool/3_3.jfif')
        # Display oil spot
        elif log_entry == 'oil_spot':
            st.subheader(text)
            space_1,space_2,space_3 = st.columns([2,5,0.3])
            with space_2:
                st.image('photo\machine\machine_4.png')
            mixer.init()
            mixer.music.load(warning)
            mixer.music.play()
            st.markdown('The part 4 is moderately damaged, please prepare the following tools to repair:')
            col_1, col_2, col_3 = st.columns(3)
            with col_1:
                st.markdown("<p style = 'text-align: center;'>Tool 4.1</p>", unsafe_allow_html=True)
                st.image('photo\photo_tool/4_1.jfif')
            with col_2:
                st.markdown("<p style = 'text-align: center;'>Tool 4.2</p>", unsafe_allow_html=True)
                st.image('photo\photo_tool/4_2.jfif')      
            with col_3:
                st.markdown("<p style = 'text-align: center;'>Tool 4.3</p>", unsafe_allow_html=True)
                st.image('photo\photo_tool/4_3.jfif')
        # Display punching_hole
        elif log_entry == 'punching_hole':
            st.subheader(text)
            space_1,space_2,space_3 = st.columns([2,5,0.3])
            with space_2:
                st.image('photo\machine\machine_5.png')
            mixer.init()
            mixer.music.load(alarm)
            mixer.music.play()
            st.markdown('The part 1 is severely damaged, please prepare the following tools to repair:')
            col_1, col_2, col_3 = st.columns(3)
            with col_1:
                st.markdown("<p style = 'text-align: center;'>Tool 1.4</p>", unsafe_allow_html=True)
                st.image('photo\photo_tool/1_4.jpg')
            with col_2:
                st.markdown("<p style = 'text-align: center;'>Tool 1.5</p>", unsafe_allow_html=True)
                st.image('photo\photo_tool/1_5.jpg')      
            with col_3:
                st.markdown("<p style = 'text-align: center;'>Tool 1.6</p>", unsafe_allow_html=True)
                st.image('photo\photo_tool/1_6.jpg')
        # Display rolled_pit
        elif log_entry == 'rolled_pit':
            st.subheader(text)
            space_1,space_2,space_3 = st.columns([2,5,0.3])
            with space_2:
                st.image('photo\machine\machine_6.png')
            mixer.init()
            mixer.music.load(alarm)
            mixer.music.play()
            st.markdown('The part 2 is severely damaged, please prepare the following tools to repair:')
            col_1, col_2, col_3 = st.columns(3)
            with col_1:
                st.markdown("<p style = 'text-align: center;'>Tool 2.4</p>", unsafe_allow_html=True)
                st.image('photo\photo_tool/2_4.jpg')
            with col_2:
                st.markdown("<p style = 'text-align: center;'>Tool 2.5</p>", unsafe_allow_html=True)
                st.image('photo\photo_tool/2_5.jpg')      
            with col_3:
                st.markdown("<p style = 'text-align: center;'>Tool 2.6</p>", unsafe_allow_html=True)
                st.image('photo\photo_tool/2_6.jpg')
        # Display silk_spot
        elif log_entry == 'silk_spot':
            st.subheader(text)
            space_1,space_2,space_3 = st.columns([2,5,0.3])
            with space_2:
                st.image('photo\machine\machine_7.png')
            mixer.init()
            mixer.music.load(alarm)
            mixer.music.play()
            st.markdown('The part 3 is severely damaged, please prepare the following tools to repair:')
            col_1, col_2, col_3 = st.columns(3)
            with col_1:
                st.markdown("<p style = 'text-align: center;'>Tool 3.4</p>", unsafe_allow_html=True)
                st.image('photo\photo_tool/3_4.jfif')
            with col_2:
                st.markdown("<p style = 'text-align: center;'>Tool 3.5</p>", unsafe_allow_html=True)
                st.image('photo\photo_tool/3_5.jfif')      
            with col_3:
                st.markdown("<p style = 'text-align: center;'>Tool 3.6</p>", unsafe_allow_html=True)
                st.image('photo\photo_tool/3_6.jfif')
        # Display waist_folding
        elif log_entry == 'waist_folding':
            st.subheader(text)
            space_1,space_2,space_3 = st.columns([2,5,0.3])
            with space_2:
                st.image('photo\machine\machine_8.png')
            mixer.init()
            mixer.music.load(alarm)
            mixer.music.play()
            st.markdown('The part 4 is severely damaged, please prepare the following tools to repair:')
            col_1, col_2, col_3 = st.columns(3)
            with col_1:
                st.markdown("<p style = 'text-align: center;'>Tool 4.4</p>", unsafe_allow_html=True)
                st.image('photo\photo_tool/4_4.jfif')
            with col_2:
                st.markdown("<p style = 'text-align: center;'>Tool 4.5</p>", unsafe_allow_html=True)
                st.image('photo\photo_tool/4_5.jfif')      
            with col_3:
                st.markdown("<p style = 'text-align: center;'>Tool 4.6</p>", unsafe_allow_html=True)
                st.image('photo\photo_tool/4_6.jfif')
            st.markdown('<ul><li>Tool 4.4</li> <li>Tool 4.5</li><li>Tool 4.6</li></ul>', unsafe_allow_html=True)
        # Display water_spot
        elif log_entry == 'water_spot':
            st.subheader(text)
            space_1,space_2,space_3 = st.columns([2,5,0.3])
            with space_2:
                st.image('photo\machine\machine_9.png')
            mixer.init()
            mixer.music.load(warning)
            mixer.music.play()
            st.markdown('The part 1 and 5 are moderately damaged, please prepare the following tools to repair:')
            col_1, col_2, col_3 = st.columns(3)
            with col_1:
                st.markdown("<p style = 'text-align: center;'>Tool 1.1</p>", unsafe_allow_html=True)
                st.image('photo\photo_tool/1_1.jpg')
            with col_2:
                st.markdown("<p style = 'text-align: center;'>Tool 5.1</p>", unsafe_allow_html=True)
                st.image('photo\photo_tool/5_1.jfif')      
            with col_3:
                st.markdown("<p style = 'text-align: center;'>Tool 5.2</p>", unsafe_allow_html=True)
                st.image('photo\photo_tool/5_2.jfif')
        # Display welding_line
        elif log_entry == 'welding_line':
            st.subheader(text)
            space_1,space_2,space_3 = st.columns([2,5,0.3])
            with space_2:
                st.image('photo\machine\machine_10.png')
            mixer.init()
            mixer.music.load(warning)
            mixer.music.play()
            st.markdown('The part 1 and 2 are moderate damaged, please prepare the following tools to repair:')
            col_1, col_2, col_3 = st.columns(3)
            with col_1:
                st.markdown("<p style = 'text-align: center;'>Tool 1.1</p>", unsafe_allow_html=True)
                st.image('photo\photo_tool/1_1.jpg')
            with col_2:
                st.markdown("<p style = 'text-align: center;'>Tool 2.2</p>", unsafe_allow_html=True)
                st.image('photo\photo_tool/2_2.jpg')      
            with col_3:
                st.markdown("<p style = 'text-align: center;'>Tool 2.3</p>", unsafe_allow_html=True)
                st.image('photo\photo_tool/2_3.jpg')

    else:

        st.header('Machine status:')
        st.subheader('The machine is working normally.')
        space_1,space_2,space_3 = st.columns([2,5,0.3])
        with space_2:
            st.image('photo\machine\machine_0.png')

elif choice == 'Future work':
    st.title('Future work')
    st.markdown('The model is working with high accuracy in predicting defect classes. However, there are always rooms for improvement such as:')
    st.markdown(""" <ul> 
                        <li>Detection using live feed video instead of uploading picture. </li>
                        <li>Deploy the demo onto website or cloud services.</li>
                        <li>Connect the demo with a database to store the defect picture as well as its timestamp for further analysis.</li>
                        <li>Apply it in practical situation.</li>
                    </ul>""", unsafe_allow_html=True)