import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow.keras.models import Model, load_model, model_from_json
from PIL import Image
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from random import choice,shuffle
from PIL import Image
from scipy import stats as st
from collections import deque,Counter
import tensorflow as tf


json_file = open("C:/Users/Shivam/Desktop/dl_project/cricket/detector_10.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
# load weights into new model
model.load_weights("C:/Users/Shivam/Desktop/dl_project/cricket/detector_10.h5")
print("Loaded model from disk")

class MaskDetect(object):
	def __init__(self):
		self.temp=True
		self.cap = cv2.VideoCapture(0)
		self.box_size = 234
		self.width = int(self.cap.get(3))
		self.rect_color = (255, 0, 0)
		# Specify the number of attempts you want.
		self.attempts = 6
		# Initially the moves will be `nothing`
		self.computer_move_name= "nothing"
		self.final_user_move = "nothing"
		self.label_names = ['0','1','2','3','4','5','6']
		# All scores are 0 at the start.
		self.computer_score, self.user_score = 0, 0
		# This variable remembers if the hand is inside the box or not.
		self.hand_inside = False
		# At each iteration we will decrease the total_attempts value by 1
		self.total_attempts = self.attempts
		# We will only consider predictions having confidence above this threshold.
		self.confidence_threshold = 0.90
		# Instead of working on a single prediction, we will take the mode of 7 predictions by using a deque object
		# This way even if we face a false positive, we would easily ignore it, by taking the mode i.e.selecting the  
		# one with maximum frequency
		self.smooth_factor = 15
		# Our initial deque list will have 'nothing' repeated 7 times.
		self.de = deque(['nothing'] * 15, maxlen=self.smooth_factor)
		# This will store the runs scored in every 6 attempts and will help in alerting user to not repeat same number of scores
		# using counter
		self.over = []
		self.count_balls = Counter(self.over)
		self.n=0
	
	def __del__(self):
		cv2.destroyAllWindows()

	def show_winner(self,user_score, computer_score):   
		print("\nTotal User Score : ",user_score)
		print("Total Computer Score : ",computer_score)
		if user_score > computer_score:
			print("\nUser won")
			img = cv2.imread("C:/Users/Shivam/Desktop/dl_project/cricket/won1.jpg")
			img = cv2.resize(img,(640,480),Image.ANTIALIAS)
			
		elif user_score < computer_score:
			img = cv2.imread("C:/Users/Shivam/Desktop/dl_project/cricket/loss1.jpg")
			img = cv2.resize(img,(640,480),Image.ANTIALIAS)
			print("\nComputer won")
		else:
			img = cv2.imread("C:/Users/Shivam/Desktop/dl_project/cricket/tie1.jpg")
			img = cv2.resize(img,(640,480),Image.ANTIALIAS)
			print("\nTie")
			
		cv2.putText(img, "Press 'ENTER' to play again, else press 'Q' for exit",
						(40, 400), cv2.FONT_HERSHEY_COMPLEX, 0.65, (34,139,34), 2, cv2.LINE_AA)
		cv2.imshow("Cricket Game", img)
			
		return True
		

	def get_frame(self):
		#temp=True
		start=0
		flag=0
		ret, frame = self.cap.read()
		frame = cv2.flip(frame, 1)
	# extract the region of image within the user rectangle
		roi = frame[5: self.box_size-5 , self.width-self.box_size + 5: self.width -5]
		roi = cv2.resize(roi,(310,310),Image.ANTIALIAS)
		roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
		roi = np.array([roi]).astype('float32') / 255.0
		roi = roi.reshape(1,310,310,1)    
		# Predict the move made
		pred = model.predict(roi)

	# Get the index of the predicted class
		move_code = np.argmax(pred[0])
		
	# Get the class name of the predicted class
		user_move = self.label_names[move_code]
		# Get the confidence of the predicted class
		prob = np.max(pred[0])
		
		cv2.putText(frame, "prediction : {} {:.2f}%".format(self.label_names[np.argmax(pred[0])], prob*100 ),
					(10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0, 0, 255), 2, cv2.LINE_AA)
		# Make sure the probability is above our defined threshold
		if prob >= self.confidence_threshold:
			
		# Now add the move to deque list from left
			self.de.appendleft(user_move)
			# Get the mode i.e. which class has occured more frequently in the last 5 moves.
			try:
				self.final_user_move = st.mode(self.de)[0][0]
			except StatisticsError:
				print('Stats error')
				
				
			if self.final_user_move != '0':
				
				if self.final_user_move != 'nothing' and self.hand_inside == False:
					
					self.over.append(self.final_user_move)
					self.count_balls = Counter(self.over)
	
					if self.count_balls[self.final_user_move]>2:
						print("\nShow different number")
						self.hand_inside = True
						cv2.putText(frame,"Show different number",
						(40, 400), cv2.FONT_HERSHEY_COMPLEX ,0.65, (255,255,255), 1, cv2.LINE_AA)
						
						# Set hand inside to True
						
							
					else:
						# This tracks the balls in the over
						self.n+=1
						print("\nBall {}".format(self.attempts-self.total_attempts+1))
						# Set hand inside to True
						self.hand_inside = True
					# Get Computer's move 
						self.computer_move_name = choice(['1','2','3','4','5','6'])

						# Display the computer's move
						#display_computer_move(computer_move_name, frame)
					
						# Subtract one attempt
						self.total_attempts -= 1
						
						# Print user and computer move
						print("User move : ",self.final_user_move)
						print("Computer move : ",self.computer_move_name)
						
						# This is a condition that will run at the end of every over, so that other conditions remain operative
						if self.n==6:
							self.over=[]
							self.n=0
													
						# Rule of game, if user move and computer move gets same, the game is terminated without results.
						if self.final_user_move == self.computer_move_name:
							#cv2.putText(frame,"Same Move, Game Terminated, System Won",
							#(160, 340), cv2.FONT_HERSHEY_COMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
							#self.user_score, self.computer_score, self.total_attempts = 0, 1, self.attempts
							#self.over=[]
							#self.n=0
							#flag=2
							self.temp=False
							self.total_attempts=0
							
						# Simultaneously we will get total score of user and computer with every attempt
						self.computer_score +=int(self.computer_move_name)
						self.user_score += int(self.final_user_move)
						# Changes the color of rectangle as long as hand is inside the rectangle frame.
						self.rect_color = (0, 250, 0)
						self.de = deque(['nothing'] * 15, maxlen=self.smooth_factor)

			if self.final_user_move == '0':           
				self.hand_inside = False
				self.rect_color = (255, 0, 0)
	
				# At the end of total attempts, it will show the results of game and ask to play again
				
				if self.total_attempts == 0 or self.temp==False:
					
					start=time.process_time()
					print("\nTotal User Score : ",self.user_score)
					print("Total Computer Score : ",self.computer_score)

					if self.temp==True:
						if self.user_score > self.computer_score:
							flag=1
							#cv2.putText(frame,"User Won",
							#	(200, 340), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
							

						elif self.user_score < self.computer_score:
							flag=2
							#cv2.putText(frame,"Computer Won",
							#	(200, 340), cv2.FONT_HERSHEY_COMPLEX ,0.7, (255,255,255), 2, cv2.LINE_AA)
						else:
							flag=3
							#cv2.putText(frame,"Tie",
							#	(200, 340), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
					elif self.temp==False:
						flag=2
		if (time.process_time()-start==500):
			self.user_score, self.computer_score, self.total_attempts = 0, 0, self.attempts

					

				
		# This is where all annotation is happening.
		if flag==1:
			cv2.putText(frame,"User Won, Refresh to play again",
					(150, 340), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
		elif flag==2:
			#self.user_score, self.computer_score, self.total_attempts = 0, 0, self.attempts
			cv2.putText(frame,"Computer Won, Refresh to play again",
					(150, 340), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
		elif flag==3:
			cv2.putText(frame,"Tie, Refresh to play again",
					(150, 340), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)				


		
		cv2.putText(frame, "Your Move: " + self.final_user_move,
						(420, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
		cv2.putText(frame, "Computer's Move: " + self.computer_move_name,
					(2, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
		cv2.putText(frame, "Your Score: " + str(self.user_score),
						(420, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
		cv2.putText(frame, "Computer Score: " + str(self.computer_score),
						(2, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
		cv2.putText(frame, "Attempts left: {}".format(self.total_attempts), (190, 400), cv2.FONT_HERSHEY_COMPLEX, 0.7,
					(0,0,0), 1, cv2.LINE_AA)
		cv2.rectangle(frame, (self.width - self.box_size, 0), (self.width, self.box_size), self.rect_color, 2)
		
		# Display the image   
		cv2.imshow("Cricket Game", frame)
		


	# Relase the camera and destroy all windows.
	
		ret, jpeg = cv2.imencode('.jpg', frame)
		return jpeg.tobytes()

	