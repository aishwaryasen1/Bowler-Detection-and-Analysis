import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import matplotlib
import operator
import pandas as pd

from random import randint

protoFile = "pose/coco/pose_deploy_linevec.prototxt"
weightsFile = "pose/coco/pose_iter_440000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

nPoints = 18
# COCO Output Format
keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 
                    'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 
                    'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
              [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
              [1,0], [0,14], [14,16], [0,15], [15,17],
              [2,17], [5,16] ]

# index of pafs correspoding to the POSE_PAIRS
# e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], 
          [19,20], [21,22], [23,24], [25,26], [27,28], [29,30], 
          [47,48], [49,50], [53,54], [51,52], [55,56], 
          [37,38], [45,46]]

colors = [ [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
         [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
         [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]
         
         # Find the Keypoints using Non Maximum Suppression on the Confidence Map
def getKeypoints(probMap, threshold=0.1):
    
    mapSmooth = cv2.GaussianBlur(probMap,(3,3),0,0)

    mapMask = np.uint8(mapSmooth>threshold)
    keypoints = []
    
    #find the blobs
    contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #for each blob find the maxima
    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

    return keypoints
    

# Find valid connections between the different joints of a all persons present
def getValidPairs(output):
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.1
    conf_th = 0.7
    # loop for every POSE_PAIR
    for k in range(len(mapIdx)):
        # A->B constitute a limb
        pafA = output[0, mapIdx[k][0], :, :]
        pafB = output[0, mapIdx[k][1], :, :]
        pafA = cv2.resize(pafA, (frameWidth, frameHeight))
        pafB = cv2.resize(pafB, (frameWidth, frameHeight))

        # Find the keypoints for the first and second limb
        candA = detected_keypoints[POSE_PAIRS[k][0]]
        candB = detected_keypoints[POSE_PAIRS[k][1]]
        nA = len(candA)
        nB = len(candB)

        # If keypoints for the joint-pair is detected
        # check every joint in candA with every joint in candB 
        # Calculate the distance vector between the two joints
        # Find the PAF values at a set of interpolated points between the joints
        # Use the above formula to compute a score to mark the connection valid
        
        if( nA != 0 and nB != 0):
            valid_pair = np.zeros((0,3))
            for i in range(nA):
                max_j=-1
                maxScore = -1
                found = 0
                for j in range(nB):
                    # Find d_ij
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    # Find p(u)
                    interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                            np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                    # Find L(p(u))
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                           pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ]) 
                    # Find E
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores)/len(paf_scores)
                    
                    # Check if the connection is valid
                    # If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair  
                    if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th :
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1
                # Append the connection to the list
                if found:            
                    valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

            # Append the detected connections to the global list
            valid_pairs.append(valid_pair)
        else: # If no keypoints are detected
#             print("No Connection : k = {}".format(k))
            invalid_pairs.append(k)
            valid_pairs.append([])
#     print(valid_pairs)
    return valid_pairs, invalid_pairs
    
# This function creates a list of keypoints belonging to each person
# For each detected valid pair, it assigns the joint(s) to a person
# It finds the person and index at which the joint should be added. This can be done since we have an id for each joint
def getPersonwiseKeypoints(valid_pairs, invalid_pairs):
    # the last number in each row is the overall score 
    personwiseKeypoints = -1 * np.ones((0, 19))

    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:,0]
            partBs = valid_pairs[k][:,1]
            indexA, indexB = np.array(POSE_PAIRS[k])

            for i in range(len(valid_pairs[k])): 
                found = 0
                person_idx = -1
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    # add the keypoint_scores for the two keypoints and the paf_score 
                    row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
    return personwiseKeypoints

# capture video
jump=[]
wrist=[]
video_path="bowl_Trim3.mp4"
cap = cv2.VideoCapture(video_path)
# Check if video file is opened successfully
if (cap.isOpened() == False):    
    print("Error opening video stream or file")

# We need to set resolutions. 
# so, convert them from float to integer. 
frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
size = (frame_width, frame_height) 
# result = cv2.VideoWriter('pose.avi',  
#                          cv2.VideoWriter_fourcc(*'MJPG'), 
#                          15, size) 


t = time.time()
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)


total = 0
count = 0
while(cap.isOpened()):
# Capture frame-by-frame
    ret, frame = cap.read()
    start = time.time()
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    if ret == True:
        inHeight = 368
        inWidth = int((inHeight/frameHeight)*frameWidth)
#         frame = cv2.resize(frame, (inHeight, inWidth), cv2.INTER_AREA)   
        # process the frame here
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        output = net.forward()
#         print(type(output))
        detected_keypoints = []
        keypoints_list = np.zeros((0,3))
        keypoint_id = 0
        threshold = 0.1

        for i in range(nPoints):
            i=1
            probMap = output[0,i,:,:]
            probMap = cv2.resize(probMap, (frame.shape[1], frame.shape[0]))
            #plt.figure()
            #plt.imshow(255*np.uint8(probMap>threshold))
            keypoints = getKeypoints(probMap,threshold)
            point0=keypoints[0][:2]
           # print(point0)
           # cv2.putText(frame, "1",(point0), cv2.FONT_HERSHEY_COMPLEX,1, (255,0,0),1)
            i=9
            probMap = output[0,i,:,:]
            probMap = cv2.resize(probMap, (frame.shape[1], frame.shape[0]))
            #plt.figure()
            #plt.imshow(255*np.uint8(probMap>threshold))
            keypoints = getKeypoints(probMap,threshold)
           # Shoulder_x=keypoints[0][0]
           # Shoulder_y=keypoints[0][1]
            Knee_y=keypoints[0][1]
           # print(Hip_y)
            i=10
            probMap = output[0,i,:,:]
            probMap = cv2.resize(probMap, (frame.shape[1], frame.shape[0]))
            #plt.figure()
            #plt.imshow(255*np.uint8(probMap>threshold))
            keypoints = getKeypoints(probMap,threshold)
           # Shoulder_x=keypoints[0][0]
           # Shoulder_y=keypoints[0][1]
            Ankle=keypoints[0][0]
            R_ank_y=keypoints[0][1]
           # print(R_ank_y)
            i=13
            probMap = output[0,i,:,:]
            probMap = cv2.resize(probMap, (frame.shape[1], frame.shape[0]))
            #plt.figure()
            #plt.imshow(255*np.uint8(probMap>threshold))
            keypoints = getKeypoints(probMap,threshold)
            L_ank_y=keypoints[0][1]
           # print(L_ank_y)
            avg_ank_y=(R_ank_y+L_ank_y)/2
         #   print(avg_ank_y)
            

            
            #print(ank_y)
          #  flag_jump=0
           # if (ank_y)>500:
              #  flag_jump=1
              #  print("flag_jump:")
               # print(flag_jump)
           # else:
            #    flag_jump=0
            
    

            i=4
            probMap = output[0,i,:,:]
            probMap = cv2.resize(probMap, (frame.shape[1], frame.shape[0]))
            #plt.figure()
            #plt.imshow(255*np.uint8(probMap>threshold))
            keypoints = getKeypoints(probMap, threshold)
            #Nose=(keypointsMapping[0], keypoints[0][:2])
          #  Wrist_x=keypoints[0][0]
           # Wrist_y=keypoints[0][1]
            R_Wrist_y=keypoints[0][1]
           # print(Wrist_y)
            n = 1 # N. . .
            #wri=[x[n] for x in Wrist]
            #print(Wrist)
            
            arm=np.subtract(Knee_y,R_Wrist_y)
            
           # arm_new=list(arm.flatten())
    

           

           
        #    if angle_deg>80:
        #        print("Bowler detected")
       #     else:
        #        print("No bowler detected")

           
            
           # print("Keypoints - {} : {}".format(keypointsMapping[i], keypoints))
            #print("Keypoints - {} : {}".format(keypointsMapping[0],keypoints[0]))
            #if keypointsMapping[part]=='Nose':
                #print(len(keypoints)) #gives number of people in video
            keypoints_with_id = []
            for i in range(len(keypoints)):
                keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                keypoint_id += 1

            detected_keypoints.append(keypoints_with_id)
#         frameClone = frame.copy()

       # print(ank_y)
       
        jump.append(avg_ank_y)
        wrist.append(arm)
       
        
     #   while flag_jump==1:
          #  store.append(arm_new)
        #    break
      #  print(store)

        
        
        for i in range(nPoints):
            for j in range(len(detected_keypoints[i])):
                cv2.circle(frame, detected_keypoints[i][j][0:2], 5, [0,255,255], -1, cv2.LINE_AA)
#         plt.figure(figsize=[15,15])
#         plt.imshow(frameClone[:,:,[2,1,0]])
        valid_pairs, invalid_pairs = getValidPairs(output)
        personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs)
        for i in range(17):
            for n in range(len(personwiseKeypoints)):
                index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                if -1 in index:
                    continue
                B = np.int32(keypoints_list[index.astype(int), 0])
                A = np.int32(keypoints_list[index.astype(int), 1])
                cv2.line(frame, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)
        fram=[]
        fra=cap.get(cv2.CAP_PROP_POS_FRAMES) #current frame
        fram.append(fra)
       # print(fra)
        
        tot=cap.get(cv2.CAP_PROP_FRAME_COUNT) #total number of frames
        #print(tot)
        fps = cap.get(cv2.CAP_PROP_FPS)
       # print(fps)

        d = {'Frame':fram,'Wrist':jump}
      #  if flag_jump==1:
           # print(d)
      #  else:
          #  print("Jump has not occurred")
        
        
        plt.figure(figsize=[15,15])
#         plt.imshow(frameClone[:,:,[2,1,0]])
        cv2.imshow('frameclone',frame)
        print("Avg. ankle position : ")
        print(jump)
        print("Wrist position : ")
        print(wrist)
        max_ank=max(jump)
        max_wrist=max(wrist)
        spot_jump = jump.index(max_ank)
        spot_wrist = wrist.index(max_wrist)
        key = cv2.waitKey(500)

        if key == 32:
            cv2.waitKey()
            print("Maximum avg. position of ankle :", max_ank)
            print("Frame number for jump:",spot_jump)
            print("Maximum position of wrist :", max_wrist)
            print("Frame number for max wrist:",spot_wrist)
        

        
            
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        count += 1
        total += (time.time()-start)
        print(count,count/total,(time.time()-start))
        
        
       
    

        

    # Break the loop
    else:        
        break
        
    
    
    

#out.release()
cap.release()
cv2.destroyAllWindows()

