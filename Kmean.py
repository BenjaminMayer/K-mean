import cv2
import numpy as np
import random
import afficher_image as aff
import morphologie as mor
import strel
def main():
    Max_L = 400
    Max_H = 400
    K =  10
    # Charger l'image et la reduire si trop grande (sinon, on risque de passer trop de temps sur le calcul...)
    imageColor = cv2.imread('perr.jpg')
    imageColor = cv2.cvtColor(imageColor, cv2.COLOR_BGR2LAB)
    #aff.afficher_image(imageColor)
    if imageColor.shape[0] > Max_H or imageColor.shape[1] > Max_L:
    	factor1 = float(Max_H)/imageColor.shape[0]
    	factor2 = float(Max_L)/imageColor.shape[1]
    	factor = min(factor1,factor2)
    	imageColor = cv2.resize(imageColor,None,fx = factor,fy= factor,interpolation = cv2.INTER_AREA)
    #E = strel.build('carre',7,None)
    #Ferm_I = mor.myClose(imageColor,E)
    #imageColor = Ferm_I-imageColor

    # Le nombre de pixels de l'image
    nb_pixels = imageColor.shape[0]*imageColor.shape[1]
    #Les coordonnees BRV de tous les pixels de l'image (les elements de E)
    bleu = imageColor[:,:,0].reshape(nb_pixels,1)
    vert = imageColor[:,:,1].reshape(nb_pixels,1)
    roug = imageColor[:,:,2].reshape(nb_pixels,1)
    #Les coordonnees BRV de chaque point-cluster (les elements de N)
    cluster_bleu = np.zeros(K)
    cluster_vert = np.zeros(K)
    cluster_roug = np.zeros(K)

    groupe = np.zeros((nb_pixels,1))
    
    for i in range(0,K):
        groupe[i,0] = i
    for i in range(K,nb_pixels):
        groupe[i,:] = random.randint(0,K-1)

    # Ici on choisit les elements aleatoirement parmi les elements de E
    for k in range(0,K):
        cluster_bleu[k] = bleu[random.randint(0,nb_pixels)]
        cluster_vert[k] = vert[random.randint(0,nb_pixels)]
        cluster_roug[k] = roug[random.randint(0,nb_pixels)]
    # Initialisation pour tester la stabilite ( calcul de la distance entre 1 point d une etape et celui de la precedante)
    cluster_bleu0 = cluster_bleu
    cluster_vert0 = cluster_vert
    cluster_roug0 = cluster_roug

    E = np.zeros(K)
    J = np.zeros(K)
    EV_bleu = np.zeros((K,nb_pixels))
    EV_vert = np.zeros((K,nb_pixels))
    EV_roug = np.zeros((K,nb_pixels))

    epsilon = 0.1
    dist = epsilon + 1
    IterMax = 10
    I = 0



    while I < IterMax or dist > epsilon:
	print I
	# 
        for i in range(0,nb_pixels):
	# Ici on calcule la distance d un element de E par rapport a tous les elements de N 
            for k in range(0, K):
                E[k] = (bleu[i,0] - cluster_bleu[k])**2 + (vert[i,0]-cluster_vert[k])**2 + (roug[i,0]-cluster_roug[k])**2
	    # On calcule le groupe auquel appartient l element de E precedant
            E = np.sqrt(E)
            groupe[i,0] = np.argmin(E)
	    # On regroupe tous les elements de E dans EV[k,:] qui ont comme groupe k
            EV_bleu[int(groupe[i,0]),i] = bleu[i,0]
            EV_vert[int(groupe[i,0]),i] = vert[i,0]
            EV_roug[int(groupe[i,0]),i] = roug[i,0]
    	# On calcule les nouveaux elements de N
        for k in range(0,K):
            cluster_bleu[k] = np.mean(EV_bleu[k,np.where(EV_bleu[k,:] != 0)])
            cluster_vert[k] = np.mean(EV_vert[k,np.where(EV_vert[k,:] != 0)])
            cluster_roug[k] = np.mean(EV_roug[k,np.where(EV_roug[k,:] != 0)])
	    # Calcul de la distance des elements de N a l etape en cours par rapport a l etape precedente
            J[k] = (cluster_bleu[k] - cluster_bleu0[k])**2 + (cluster_vert[k]-cluster_vert0[k])**2 + (cluster_roug[k]-cluster_roug0[k])**2
        J = np.sqrt(J)
        dist = np.sum(J)
        cluster_bleu0 = cluster_bleu
        cluster_vert0 = cluster_vert
        cluster_roug0 = cluster_roug
        I = I+1
  #On change le format de groupe afin de le rammener au format de l'image d'origine
    groupe = np.reshape(groupe, (imageColor.shape[0],imageColor.shape[1]))

    #On change chaque pixel de l'image selon le cluster auquel il appartient
    #Il doit prendre comme nouvelle valeur la position moyenne du cluster
    for i in range(0,imageColor.shape[0]):
        for j in range(0,imageColor.shape[1]):
            imageColor[i,j,0] = (cluster_bleu[int(groupe[i,j])])
            imageColor[i,j,1] = (cluster_vert[int(groupe[i,j])])
            imageColor[i,j,2] = (cluster_roug[int(groupe[i,j])])
    #imageColor = cv2.cvtColor(imageColor, cv2.COLOR_LAB2BGR)
    aff.afficher_image(imageColor)
    cv2.imwrite('perrk10Lab.png', imageColor) 

if __name__ == "__main__":
    main()
