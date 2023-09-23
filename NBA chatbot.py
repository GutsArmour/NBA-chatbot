# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 19:58:52 2023

@author: ahmed
"""
import pandas
import os
import aiml
from sympy.assumptions import ask
import csv
import numpy as np
from io import BytesIO
from bs4 import BeautifulSoup

import wikipedia
import json, requests
from nltk.sem import Expression
from nltk.inference import ResolutionProver
import pyttsx3
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import tensorflow as tf
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from PIL import Image

read_expr = Expression.fromstring

kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="NBAlogic.xml")

def openKB(kbList, kbPath):
    if os.path.isfile(kbPath):
        data = pandas.read_csv(kbPath, header = None)
        [kbList.append(read_expr(row)) for row in data[0]]
        if ResolutionProver().prove(read_expr("barcelona" + "(nba_team)"), kbList):
            print("Error in the knowledgebase")
            return False
        return True
    else:
        print("Knowledgebase file not found!")
        return False

def aimlAnswer(userInput):
    return kern.respond(userInput)

def wikipediaFunc(params):
    try:
        wSummary = wikipedia.summary(params[1], sentences=3,auto_suggest=False)
        return (wSummary)
    except:
        return "Sorry, I do not know that. Be more specific!"
    
def weatherFunc(params):
    succeeded = False
    api_url = r"http://api.openweathermap.org/data/2.5/weather?q="
    APIkey = "5403a1e0442ce1dd18cb1bf7c40e776f"
    response = requests.get(api_url + params[1] + r"&units=metric&APPID="+APIkey)
    
    if response.status_code == 200:
        response_json = json.loads(response.content)
        if response_json:
            t = response_json['main']['temp']
            tmi = response_json['main']['temp_min']
            tma = response_json['main']['temp_max']
            hum = response_json['main']['humidity']
            wsp = response_json['wind']['speed']
            wdir = response_json['wind']['deg']
            conditions = response_json['weather'][0]['description']
            return f"The temperature is {t}°C, varying between {tmi} and {tma} at the moment, humidity is {hum}%, wind speed {wsp}m/s, and conditions are {conditions}."
            succeeded = True
    if not succeeded:
        return "Sorry, I could not resolve the location you gave me."

def addStatement(params, kbList, kb):
    object,subject=params[1].split(' is ')
    object,subject=params[1].split(' is ')
    if subject.startswith("a "):
        subject = subject[2:]
    elif subject.startswith("an "):
        subject = subject[3:]
    subjectFormat = subject.replace(' ', '_').lower()
    objectFormat = object.replace(' ', '_').lower()
    expr=read_expr(subjectFormat + '(' + objectFormat + ')')
    
    if ResolutionProver().prove(expr, kbList):
        return "This statement is already in my database"
    elif ResolutionProver().prove(not(expr), kbList):
        return "This statement contradicts my database"
    else:
        kbList.append(expr)
        with open('kb.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([str(expr)])
        return f'OK, I will remember that {object} is a {subject}'

def checkStatement(params, kbList):
    object,subject=params[1].split(' is ')
    if subject.startswith("a "):
        subject = subject[2:]
    elif subject.startswith("an "):
        subject = subject[3:]
        
    subjectFormat = subject.replace(' ', '_').lower()
    object = object.strip()
    objectFormat = object.replace(' ', '_').lower()
    expr=read_expr(subjectFormat + '(' + objectFormat + ')')
    answer=ResolutionProver().prove(expr, kbList, verbose=False)
    
    found_subject = False
    found_object = False
    for expr in kbList:
        if isinstance(expr, str):
            if subjectFormat in expr.lower():
                found_subject = True
            if objectFormat in expr.lower():
                found_object = True
        elif isinstance(expr, Expression):
            if subjectFormat in str(expr).lower():
                found_subject = True
            if objectFormat in str(expr).lower():
                found_object = True
    if found_subject and found_object:
        pass
    elif not found_subject and not found_object:
        return "Sorry, I don't know anything about " + subject + " and " + object + "."
    elif not found_subject:
        return "Sorry, I don't know anything about " + subject + "."
    else:
        return "Sorry, I don't know anything about " + object + "."
   
    if answer is True:
        return "Correct."
    elif answer is not True:
        return "This is incorrect."   
        
def knowledgeFunc(kbList):
    if not kbList:
        print("I don't know anything yet.")
    else:
        print("Here's what I know:")
        for statement in kbList:
            if str(statement).startswith("nba_team"):
                arguments = str(statement).split("(")[1].split(")")[0].split("_")
                if arguments[0] != arguments[1]:
                    team_name = " ".join(arguments)
                    print (team_name + " is an NBA team")
            if str(statement).startswith("wnba_team"):
                arguments = str(statement).split("(")[1].split(")")[0].split("_")
                if arguments[0] != arguments[1]:
                    team_name = " ".join(arguments)
                    print (team_name + " is an WNBA team")
        
def getClass(modelFile):
    model = load_model(modelFile)
    inputSize = (216, 216) 
    root = tk.Tk()
    root.withdraw()
    root.update()        
    file_path = filedialog.askopenfilename()
    
    img = load_img(file_path, target_size=inputSize)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    
    probs = model.predict(img)    
    threshold = 0.2
    class_indices = np.where(probs > threshold)[1]
    
    class_names = {0: "baseball", 1: "basketball", 2: "beachball", 3: "billiard ball", 4: "bowling ball", 5: "brass", 6: "buckeyball", 7: "cannon ball", 8: "crochet ball", 9: "cricket ball", 10: "crystal ball", 11: "eyeball", 12: "football", 13: "golf ball", 14: "marble", 15: "meat ball", 16: "medicine ball", 17: "paint ball", 18: "pokemon ball", 19: "puffball", 20: "rubberband ball", 21: "screwball", 22: "sepak takraw ball", 23: "soccer ball", 24: "tennis ball", 25: "tether ball", 26: "volley ball", 27: "water polo ball", 28: "wiffle ball", 29: "wrecking ball"}
    class_names_found = [class_names[i] for i in class_indices]
    if len(class_names_found) > 1:
        class_names_str = ', '.join(class_names_found[:-1]) + ' and ' + class_names_found[-1]
    else:
        class_names_str = class_names_found[0]  
    return f"This image contains a {class_names_str}."

def searchImg(params):
    query = params[1]
    url = f"https://www.google.com/search?q={query}&tbm=isch"
    
    response = requests.get(url)
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    imgUrl = soup.find_all('img')[1]['src']
    
    response = requests.get(imgUrl)
    img = Image.open(BytesIO(response.content))
    
    img.show()
     
def bestMatch(userInput):
    with open("QnA.csv", "r", newline="\n") as csvFile:
        csvReader = csv.reader(csvFile, delimiter=",")
        csvSentences = [row[0] for row in csvReader]

    csvSentences.append(userInput)
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidfVectors = vectorizer.fit_transform(csvSentences)
    cosSims = cosine_similarity(tfidfVectors[-1], tfidfVectors[:-1]).flatten()
    bestMatchIndex = cosSims.argmax()

    if cosSims[bestMatchIndex] > 0:
        with open("QnA.csv", "r", newline="\n") as csvFile:
            csvReader = csv.reader(csvFile, delimiter=",")
            for i, row in enumerate(csvReader):
                if i == bestMatchIndex:
                    return row[1]
    return None

def getTextInput():
    return input("> ")

def getVoiceInput():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something!")
        audio = r.listen(source)
        userInput = r.recognize_google(audio)
    return userInput

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    
def main():
    kbList = []
    kb = "kb.csv"
    modelFile = "ballsV3.h5"
    openKB(kbList, kb)
    
    while True:
        try:
            userChoice = input("Enter 't' for text input or 'v' for voice input: ")
            if userChoice == 't':
                userInput = getTextInput().lower()
            elif userChoice == 'v':
                userInput = getVoiceInput()
            else:
                print("Invalid input. Please enter 't' or 'v'.")
                continue
        except (KeyboardInterrupt, EOFError) as e:
            print("Bye!")
            break
        except sr.UnknownValueError:
            print("Sorry, I didn't understand that.")
            continue
        
        # pre-process user input and determine response agent (if needed)
        responseAgent = 'aiml'
        # activate selected response agent
        if responseAgent == 'aiml':
            answer = aimlAnswer(userInput)
            
        # post-process the answer for commands
            if answer[0] == '#':
                params = answer[1:].split('$')
                cmd = int(params[0])
                
                if cmd == 0:
                    print(params[1])
                    break
                    if userChoice == 'v':
                        speak(params[1])
                        break
                        
                elif cmd == 1:
                    print(wikipediaFunc(params))
                    if userChoice == 'v':
                        speak(wikipediaFunc(params))
                        
                elif cmd == 2:
                    print(weatherFunc(params))
                    if userChoice == 'v':
                        speak(weatherFunc(params))
                        
                elif cmd == 31:
                    print(addStatement(params, kbList, kb))
                    if userChoice == 'v':
                        speak(addStatement(params, kbList, kb))
                        
                elif cmd == 32:
                    print(checkStatement(params, kbList))
                    if userChoice == 'v':
                        speak(checkStatement(params, kbList))
                        
                elif cmd == 33:
                    knowledgeFunc(kbList)
                    
                elif cmd == 34:
                    result = getClass(modelFile)
                    if userChoice == "t":
                        print(result)
                    if userChoice == 'v':
                        print(result)
                        speak(result)
                        
                elif cmd == 35:
                    searchImg(params)
                    
                elif cmd == 99:
                    qna_response = bestMatch(userInput)
                    if qna_response:
                        print(qna_response)
                        if userChoice == 'v':
                            speak(qna_response)
                    else:
                        print("Sorry, I did not get that")
            else:
                print(answer)
                if userChoice == "v":
                    speak(answer)
            
            
if __name__ == '__main__':
    main()
