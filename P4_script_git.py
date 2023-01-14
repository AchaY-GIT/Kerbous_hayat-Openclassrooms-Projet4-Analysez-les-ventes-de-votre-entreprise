#!/usr/bin/env python
# coding: utf-8

# In[855]:



# import des librairies dont nous aurons besoin
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import datetime as dt
import scipy.stats as st


# In[856]:


# chargement et affichage des données


custumers= pd.read_csv('D:\\Aopenclassroom\\P4\\dataset_P4/customers.csv')
products= pd.read_csv('D:\\Aopenclassroom\\P4\\dataset_P4/products.csv')
transactions= pd.read_csv('D:\\Aopenclassroom\\P4\\dataset_P4/transactions.csv')

lignes et colonnes
# In[857]:



custumers.shape


# In[858]:


products.shape


# In[859]:


transactions.shape


# colonnes

# In[860]:



custumers.columns


# In[861]:


products.columns


# In[862]:


transactions.columns

les informations
# In[863]:


custumers.info


# In[864]:


custumers.value_counts()


# In[865]:


custumers.sort_values('client_id', ascending = False).head()


# In[866]:


custumers.dtypes


# In[867]:


#on remarque ici la presence d'une forme differente de client_id ct relatives aux tests qu'on va supprimer 


custumers.drop(custumers[custumers.client_id.str.startswith("ct_") == True].index, inplace=True)


# In[868]:


custumers.sort_values('client_id', ascending = False).head()


# In[869]:


products.info


# In[870]:


transactions.info

entete
# In[871]:


custumers.head(5).T


# In[872]:


custumers.tail(10).T


# In[873]:


products.head(5).T


# In[874]:


products.tail(10).T


# In[875]:


transactions.head(5).T


# In[876]:


transactions.tail(10).T

Recherche des valeurs manquantes
# In[877]:


products.isna().sum()


# In[878]:


products.isna().sum()


# In[879]:


transactions.isna().sum()


# In[880]:


products.describe().T


# In[881]:


#ici on remarque qu'il exist des valeurs negatives dans le price ce qui n'est pas logique

#on cherche les autres valeur négatives 

products[products.price <= 0]


# In[882]:



#Suppression de la ligne ou les prix sont négatifs
products.drop(products[products.price <= 0].index, inplace=True)
products.head()


# In[883]:


#verifier la coherence des données sur le produit 
products.describe().T


# In[884]:



transactions.loc[transactions.id_prod.duplicated(keep=False),:]

les données semblent logiques 
# In[885]:



#Vérifier la cohérence de l'âge des clients
custumers.describe().T


# In[886]:


custumers[custumers.birth == 2004.0]


# In[887]:


print(transactions.date.max())
print(transactions.date.min())


# In[888]:


transactions.sort_values('date', ascending = False).head()


# In[889]:


transactions.client_id.describe()


# In[890]:


transactions.sort_values('date', ascending = False).tail(15).head()


# In[891]:


products.head()


# In[892]:


products[['price']].boxplot(figsize=(20,10))
#plt.savefig("D:\\Aopenclassroom\\P4\\p4_graphe")


# In[ ]:





# In[893]:


sns.set_style("whitegrid")
sns.boxplot(data=products,y = 'price', palette="deep")
sns.despine(left=True)


plt.title('Dispersion des prix des produits')
plt.boxplot
plt.savefig("p4_figures/box_dis_prix_produits.png")
plt.ylim(0, 50)


# In[894]:


custumers.isnull().sum()


# In[895]:


products.isnull().sum()


# In[896]:


transactions.isnull().sum()


# In[897]:


custumers.loc[custumers['client_id'].duplicated(keep=False),:]


# In[898]:


products.loc[products['id_prod'].duplicated(keep=False),:]


# In[899]:


transactions.loc[transactions['session_id'].duplicated(keep=False),:]


# In[900]:


transactions.sort_values('date', ascending = False).head()

#on remarque que les données T_0 sont des données tests qu'on doit enlever 

# In[901]:


#suppression des données test
transactions.drop(transactions[transactions.id_prod == 'T_0'].index, inplace=True)


# In[902]:


transactions.sort_values('date', ascending = False).head()


# In[903]:


transactions.sort_values('date', ascending = False).tail()


# In[904]:


#Convetir les valeurs de la colonne 'date' en format date 

transactions['date'] = pd.to_datetime(transactions.date, format='%Y-%m-%d %H:%M:%S', errors = 'coerce')

#Vérification
transactions.dtypes


# In[905]:


transactions.date.value_counts()


# In[906]:


transactions.info()


# In[907]:


transactions.date.value_counts()


# In[908]:


#suppression de données date depassant la date actuelle  

transactions.drop(transactions[transactions.date > '2021-10-30'].index).head()


# In[909]:


transactions.sort_values('date', ascending = False).head()


# In[910]:


transactions.sort_values('date', ascending = False).tail()


# # mission 2

# In[911]:


#je commence par l analyse des clients


# In[912]:



# Diagramme en secteur
custumers["sex"].value_counts(normalize=True).plot(kind='pie')
# Cette ligne assure que le pie chart est un cercle plutôt qu'une éllipse
plt.axis('equal') 
plt.title('répartition du sex des client')
plt.savefig("p4_figures/dis_s_pie_client.png")
plt.show() # Affiche le graphique

# Diagramme en tuyaux d'orgues
custumers["sex"].value_counts(normalize=True).plot(kind='bar')
plt.title('dispertion du sex des client')
plt.savefig("p4_figures/s_s_bat_client.png")
plt.show()

des indicateurs de tendance centrale et de dispersion 
# In[913]:


#une mesure de tendance centrale 
products['price'].mode()


# In[914]:


#la mediane du prix ds produits
products['price'].median()


# In[915]:


#distribution de l'age des clients
# Histogramme 

custumers["birth"].hist(density=True)
plt.title('distribussion de l\'age des clients')
plt.savefig("p4_figures/dis_age_clients.png")
plt.show()


# # ventes par categories

# In[916]:


#Visualisation des ventes par produit
price_categ=products.groupby('categ').sum().reset_index()
price_categ


# # Analyser les prix des transactions

# # Analyser les ventes 

# In[917]:


#on join les deux tables transaction et client
#union ds deux tables 
trans_prod= pd.merge(transactions,products, on='id_prod', how='left')

trans_prod.head()


# In[918]:


trans_prod_n=trans_prod.drop(trans_prod[trans_prod.date>'2021-10-30'].index)
trans_prod_n.head(5)


# In[919]:


#faire l'histograme des ventes journalières 


# In[920]:


#vérifier des valeurs manquantes


trans_prod_n.isnull().sum()


# In[921]:


#reperer les produits eyant des valeurs manquantes

na=trans_prod_n.isnull().sum()
na[na>0]

On observe que notre jeu de données contient 70 valeurs manquantes. Regardons les plus en détails :
# In[922]:


trans_prod_n.loc[trans_prod_n['price'].isnull(),:].head()


# In[923]:


#on remarque que le produit manquant est celui eyant un id_prod=0_2245 mais pour etre plus sur on place les valeurs manquantes dans un dataframe 


# In[924]:


trans_prod_n_na = trans_prod_n.loc[trans_prod_n['price'].isnull(),:]
trans_prod_n_na.head()


# In[925]:


trans_prod_n.head()


# In[926]:


#on recupere les id_prod des produits manquants

trans_prod_n_na.id_prod.unique()

on verifie le prix de ce produit 
# In[927]:


products.loc[products.id_prod=='0_2245']

#le seul poroduit representant des pris et categ manquants est le id_prod=0_2245

il commence par 0 donc il apartien a la categorie 0 on le remplace par la myenne des prix de cette categorie 0  
# In[928]:


trans_prod_0=trans_prod_n[trans_prod_n.categ==0.0]
trans_prod_0.head()


# In[929]:


mean_categ0=trans_prod_0.price.mean()
mean_categ0


# In[930]:




trans_prod_n['price'] = trans_prod_n.price.fillna(mean_categ0)


# In[931]:




trans_prod_n.loc[trans_prod_n.id_prod=='0_2245'].head()


# In[932]:


trans_prod_n['price'] = trans_prod_n['price'].apply(lambda x: round(x, 2))


# In[933]:


trans_prod_n['categ'] = trans_prod_n.categ.fillna(0.0)


# In[934]:


trans_prod_n.loc[trans_prod_n.id_prod=='0_2245'].head()


# In[935]:


#verifier les doublans 
trans_prod_n.loc[trans_prod_n[['session_id']].duplicated(keep=False),:].head()


# In[936]:


db=trans_prod_n.loc[trans_prod_n[['session_id']].duplicated(keep=False),:]


# In[937]:


#pas de valeurs manquantes


# In[938]:


#verifier les nan
trans_prod_n.isna().sum()


# In[939]:


# pas de nan 


# In[ ]:





# des indicateurs de tendance centrale et de dispersion

# In[940]:


#dans cet analyse je ne prend pas en compte le mois d octobre vous trouverez l explication lors de l analyse du chiffre d'affaire  


# In[941]:


trans_prod_net=trans_prod_n.drop(trans_prod_n[trans_prod_n.date>'2021-09-30'].index)
trans_prod_net.head()


# In[942]:


#une mesure de tendance centrale 
trans_prod_net['price'].mode()

ici on parle d'une distribution unimodalele on a un seul pic, qui est le prix le plus repondu est 15.99E 
# In[943]:


#la mediane du prix des produits vendues
trans_prod_net['price'].median()


# In[944]:


trans_prod_net['price'].mean()


# In[945]:


#distribussion des prix des produits vendus
# Histogramme
trans_prod_net["price"].hist(density=True)
plt.title('distribussion des prix des produits vendus')
plt.show()
# Histogramme pour les produits inferieurs a 100E
trans_prod_net[trans_prod_net.price < 50]["price"].hist(density=True,bins=20)
plt.savefig("p4_figures/dis_prod_ventes.png")
plt.title('distribussion des prix des produits vendus')
plt.show()
#reduire la taille 10, 5

On remarque que les produits les plus vendues sont ceux inferieurs a 20E
# ## les mesures de dispersion

# In[946]:


#calculer la variance
round(trans_prod_net['price'].var(),2)


# In[947]:


# la variance empirique corrigée 
round(trans_prod_net['price'].var(ddof=0),2)


# In[948]:


#calculer l'écart-type de la variable price
round(trans_prod_net['price'].std(ddof=0),2)

#les valeurs sont à l equart de 18e trés dispersse par rapport à la moyenne 
#interval de confiance ?
# In[949]:


#construire une boîte à moustaches
trans_prod_net.boxplot(column="price", vert=False,figsize=(10,5))
plt.xlim(0,35)
plt.savefig("p4_figures/evolution_ventes.png")
plt.show()
#rajouter la moyenne

#beaucoup de donnee> a la moyenne donc c'est disperssé
# # #les mesures de forme

# In[950]:


#calcul du skewness
round(trans_prod_net['price'].skew(),2)

c est loin de >0 donc une asymetrie + beaucoup de données tres elevees VERS LE MAX par raport a la mediane 
# In[951]:


#calcul du kurtosis
round(trans_prod_net['price'].kurtosis(),2)

+ valeurs ont un pic tres important pour des prix faibles 
# ## les mesures de concentration

# In[952]:


#la courbe de Lorenz :
ech3 = trans_prod_net['price'].values

n3 = len(ech3)
lorenz3 = np.cumsum(np.sort(ech3)) / ech3.sum()
lorenz3 = np.append([0],lorenz3) # La courbe de Lorenz commence à 0

xaxis3 = np.linspace(0-1/n3,1+1/n3,n3+1) #Il y a un segment de taille n pour chaque individu, plus 1 segment supplémentaire d'ordonnée 0. Le premier segment commence à 0-1/n, et le dernier termine à 1+1/n.

#rajouter la courbe Y=X pour comparer 

plt.plot(np.linspace(0,1,len(lorenz3)), lorenz3, drawstyle='steps-post', color='blue', label='Lorenz')

plt.plot([0, 1], [0, 1], 'r-', lw=2, label='Distribution égalitaire')
plt.vlines(x=.76, ymin=0, ymax=.5, color='darkgreen', linestyle='--', linewidth=1, label='Mediane')
plt.hlines(xmin=.76, xmax=0, y=.5, color='darkgreen', linestyle='--', linewidth=1)

plt.title('Courbe de Lorenz des prix de vente')
plt.xlabel("Distribution des ventes (%)")
plt.ylabel("Cumul des prix de ventes (%)")
plt.legend(loc="best")
plt.savefig("p4_figures/courbe_lorenz.png")

plt.show()


# In[953]:


#l'indice de Gini
AUC3 = (lorenz3.sum() -lorenz3[-1]/2 -lorenz3[0]/2)/n3 # Surface sous la courbe de Lorenz. Le premier segment (lorenz[0]) est à moitié en dessous de 0, on le coupe donc en 2, on fait de même pour le dernier segment lorenz[-1] qui est à moitié au dessus de 1.
S3 = 0.5 - AUC3 # surface entre la première bissectrice et le courbe de Lorenz
gini3 = 2*S3
gini3
print(round(gini3 ,2))


# In[954]:


#il ya une inegalite dans la distribution des ventes tq la moitier des ventes sont realisé avec 80% 
#des produits (les moin chers) et cela est normale  


# # Analyse du chiffre d'affaire 

# In[955]:


#celle ci justifie pour quoi j ai du supprimer le mois d'octobre 


# In[958]:


chifr_afr=trans_prod_n.groupby(trans_prod_n['date'].dt.strftime('%Y-%m'))['price'].sum().reset_index()
#.sort_values()
chifr_afr


# In[959]:


ordered_chifr_afr=chifr_afr.sort_values('date', ascending=True)
ordered_chifr_afr


# In[960]:


#graph évolution des ventes durant les mois de cette année
ordered_chifr_afr.plot(x='date', y='price')

plt.title('ventes mensuelles')
plt.xlabel('mois')
plt.ylabel('price')
plt.savefig("p4_figures/evolution_ventes_ment.png")

plt.show()

on remarque une forte baisse des ventes le mois de septembre et octobre
###il faut aller jour par jour
# In[961]:


transactions[(transactions.date<'2021-10-01')&(transactions.date>'2021-09-01') ].value_counts()

On remarque 33266 comandes en mois de septembre 
# In[962]:


transactions[(transactions.date<'2021-09-01')&(transactions.date>'2021-08-01')].value_counts()

On remarque que le nembre de commandes sont presque identiques aux comandes precedentes mais le chifre d afaire a baisser 

# In[963]:


products.info()


# In[964]:


#Visualisation les ventes par produit
price_categ=products.groupby('categ').sum().reset_index()
price_categ


# In[965]:


trans_oct= trans_prod_n[trans_prod_n.date>'2021-09-30']
trans_oct.head()


# In[966]:


#Visualisation les ventes par produit
oct_categ=trans_oct.groupby('categ').sum().reset_index()
oct_categ


# In[967]:


#on remarque que la categorie dont le chiffre d'affaire a baissé est la categorie n 2 


# In[968]:


trans_oct_j=trans_oct.groupby(trans_oct['date'].dt.strftime('%Y-%m-%d'))['price'].sum().reset_index()
#.sort_values(by='date',ascending=False)
trans_oct_j.head(3)


# In[969]:


#graph évolution des ventes durant les jours du mois d octobre 
trans_oct_j.plot(x='date', y='price',figsize=(10,5))

plt.title('ventes journalières')
plt.xlabel('jours')
plt.ylabel('price')
plt.savefig("p4_figures/evolution_ventes_jr.png")

plt.show()


# #on remarque une baisse importante en une seule journnée dés le debut du mois 
# qui c est ensuite stagnée, c'est pourquoi j'ai décidé de supprimer l'enssemble du mois d'octobre 

# In[970]:


transactions.drop(transactions[transactions.date>'2021-09-30'].index)
trans_prod_net=trans_prod_n.drop(trans_prod_n[trans_prod_n.date>'2021-09-30'].index)
trans_prod_net.head()


# In[971]:


chifr_afr_net=trans_prod_net.groupby(trans_prod['date'].dt.strftime('%Y-%m'))['price'].sum().reset_index()
chifr_afr_net


# In[972]:


ordered_chifr_afr_net=chifr_afr_net.sort_values('date', ascending=True)
ordered_chifr_afr_net


# In[973]:


#graph évolution des ventes durant les mois de cette année
ordered_chifr_afr_net.plot(x='date', y='price',figsize=(10,5))

plt.title('ventes mensuelles')
plt.xlabel('mois')
plt.ylabel('price')
plt.savefig("p4_figures/evolution_chfr_afr.png")

plt.show()


# In[ ]:





# # analyser les mesures de tendance centrale

# In[974]:



chifr_afr_net['price'].mode()


# In[ ]:




on remarque plusieurs pic on est sur une  distribution: plurimodale.
# In[975]:


#la moyenne
moyenne_prix=chifr_afr_net['price'].mean()
print('la moyenne est :',round(moyenne_prix, 2),'€')


# In[976]:


#La médiane
médiane_prix=chifr_afr_net['price'].median()
print('la medianne est :',round(médiane_prix,2),'€')


# # les mesures de dispersion 

# In[ ]:





# In[977]:


#calculer la variance
round(chifr_afr_net['price'].var(),2)


# In[978]:


# la variance empirique corrigée 
round(chifr_afr_net['price'].var(ddof=0),2)


# In[979]:


#calculer l'écart-type de la variable price
round(chifr_afr_net['price'].std(ddof=0),2)


# In[980]:


#construire une boîte à moustaches

sns.boxplot(x ='price', data = trans_prod_n, palette="Set1")
plt.title('Dispersion des prix de vente ')

plt.savefig("p4_figures/box_dis_prix_ventes.png")
plt.show()


# les mesures de forme

# In[981]:


#calcul du skewness
round(chifr_afr_net['price'].skew(),2)

γ1>0 alors la distribution est étalée à droite.
# In[982]:


#calcul du kurtosis
round(chifr_afr_net['price'].kurtosis(),2)

γ2>0 , alors elle est moins aplatie que la distribution normale : les observations sont plus concentrées.
# # les mesures de concentration

# # analyser les ventes par categories

# In[983]:



#analyse sectoriel
# Diagramme en secteurs
trans_prod_net["categ"].value_counts(normalize=False).plot(kind='pie')
# Cette ligne assure que le pie chart est un cercle plutôt qu'une éllipse
plt.axis('equal') 
plt.savefig("p4_figures/dis_prix_cat_pie.png")
plt.title('distribution des ventes par catégories')
plt.show() # Affiche le graphique

# Diagramme en tuyaux d'orgues
trans_prod_net["categ"].value_counts(normalize=True).plot(kind='bar')
plt.title('distribution des ventes par catégories')
plt.savefig("p4_figures/dis_prix_cat.png")
plt.show()

#on remarque que la categorie 0 prend 60% des vente contre 30% par la categorie 1 et seulement 10% de
#par la categorie 2.
on supose que les prix des produits de categorie 2 sont trés élevés comparé aux autres categories
ce qui nécessite une analyse
# # analyse des prix par categories

# In[984]:


trans_prod_net["categ"].unique()


# In[985]:


for cat in trans_prod_net["categ"].unique():
    subset = trans_prod.loc[trans_prod.categ == cat, :] # Création du sous-échantillon
    print("-"*20)
    print(cat)
    print("moy:\n",subset['price'].mean())
    print("med:\n",subset['price'].median())
    print("mod:\n",subset['price'].mode())
    subset["price"].hist() # Crée l'histogramme
    plt.savefig("p4_figures/anal_prix_categ.png")
    plt.title('distribution des prix par categories')
    plt.show()

les prix varient de >0 à 20 pour categ 0 et de 10 à 30 pour categ 1 
mais pour la categ 2 ils depassent 50€ (entre 50 et 70) 
aprés ils se désperssent et s'étalent vers la droite  
# In[986]:


tran_price_categ=trans_prod.groupby("categ")['price'].sum().reset_index()
tran_price_categ


# In[987]:


#boite à moustache
trans_prod_net.boxplot(column='price', by='categ', showmeans=True)


plt.title('Dispersion des prix par catégorie')
plt.xlabel('Catégories')
plt.ylabel('prices of products')


plt.show()


# In[988]:



sns.boxplot(x = 'categ', y = 'price', data = trans_prod_net, palette="deep")
plt.title('Dispersion des prix de vente par catégories')
plt.ylim(0,100)
plt.savefig("p4_figures/Dis_prix_vent_catégorie.png")
plt.show()


# In[ ]:





# # l’analyse bivariée entre categories et prix

# ## Analysez une variable quantitative (price) et une qualitative (categorie) par ANOVA

# In[989]:


X = "categ" # qualitative
Y = "price" # quantitative


sous_echantillon_cat = trans_prod_net.copy()


# In[990]:


modalites_cat = sous_echantillon_cat[X].unique()
groupes_cat = []
for m in modalites_cat:
    groupes_cat.append(sous_echantillon_cat[sous_echantillon_cat[X]==m][Y])

# Propriétés graphiques (pas très importantes)    
medianprops = {'color':"black"}
meanprops = {'marker':'o', 'markeredgecolor':'black',
            'markerfacecolor':'firebrick'}
    
plt.boxplot(groupes_cat, labels=modalites_cat, showfliers=False, medianprops=medianprops, 
            vert=False, patch_artist=True, showmeans=True, meanprops=meanprops)
plt.savefig("p4_figures/box_Dis_prix_vent_catég.png")
plt.title('distribution des prix par catégories')
plt.show()

les moyennes sont differentes 
On voit ici que les montants sont très différents d'une catégorie2  à l'autre
# ###  calcule de  𝜂2  (eta carré ou eta squared).

# In[991]:


X = "categ" # qualitative
Y = "price" # quantitative

sous_echantillon_mois = chifr_afr_net["price"].copy()


# In[992]:


X = "categ" # qualitative
Y = "price" # quantitative

def eta_squared(x,y):
    moyenne_y = y.mean()
    classes = []
    for classe in x.unique():
        yi_classe = y[x==classe]
        classes.append({'ni': len(yi_classe),
                        'moyenne_classe': yi_classe.mean()})
    SCT = sum([(yj-moyenne_y)**2 for yj in y])
    SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes])
    return SCE/SCT
    
round(eta_squared(sous_echantillon_cat[X],sous_echantillon_cat[Y]),2)

#il ya un lien entre les variable et la categorie 2 qui est tres eleve  On obtient un résultat proche de 1, ce qui laisse penser qu'il y a effectivement une corrélation entre la categorie et les prix des produits c'est ce que nous avions observé sur le graphique en haut.
# In[ ]:





# In[993]:


#il faut d'abord faire le fit des données
import statsmodels.formula.api; fit = statsmodels.formula.api.ols('price ~ categ', data = sous_echantillon_cat).fit()


# In[994]:


#on calcule l'anova : 
import statsmodels.api; table = statsmodels.api.stats.anova_lm(fit)
table 


# In[995]:


#utilise une correction pour l'hétéroscédasticité
statsmodels.stats.anova.anova_lm(fit, robust = 'hc3')


# In[996]:


#Tukey HSD après une ANOVA
#res = statsmodels.stats.multicomp.pairwise_tukeyhsd(yValues, xValues, alpha = 0.01) 


# In[ ]:





# # analyse de l'age

# In[997]:


#on unifie la transaction et custumers
trans_cust=pd.merge(transactions,custumers)
trans_cust.head()


# In[998]:


#verifier les valeurs manquantes
trans_cust.isnull().sum()


# In[999]:


#verifier les na
trans_cust.isna().sum()


# In[1000]:


trans_cust.loc[trans_cust.duplicated(keep=False),:]


# In[1001]:


#pas de doublants 


# In[1002]:


#detecter les outliers
trans_cust.describe()


# In[1003]:


trans_cust['age']=2021-trans_cust['birth']
trans_cust


# In[1004]:


trans_cust['age'].describe()


# ## Représentez la distribution empirique de la variable age

# In[1005]:


# Histogramme
trans_cust["age"].hist(density=True)
plt.savefig("p4_figures/evolution_ventes.png")
plt.savefig("p4_figures/Dis_age.png")
plt.title('distribution empirique de la variable age')
plt.show()


# In[ ]:





# In[1006]:


n=len(trans_cust.client_id)
n


# In[1007]:


#determiner les classes avec la règle de Sturges (1926)
import math
1 + math.log2(n)


# In[1008]:



tranch_age=pd.DataFrame(trans_cust)


#tranch_age['trch_age'] = pd.cut(trans_cust['age'], [17, 30,40,50,92])

tranch_age['trch_age'] = pd.cut(trans_cust['age'],5)
tranch_age=tranch_age.sort_values('age', ascending=True)
tranch_age.head()


# In[1009]:


# Diagramme en secteurs
tranch_age["trch_age"].value_counts(normalize=True, sort=False).plot(kind='bar',width=0.5)
plt.title('distribution des ventes par tranches d\'age')
plt.savefig("p4_figures/bat_dis_age.png")

on remarque que la majorité des transactions sont faite par une tranche 32 à 47 ans 
# ## Présentez une variable age sous forme de tableau

# In[1010]:


#je rajoute une colonne pour separer les tranche d'age


# In[1011]:


effectifs_age = tranch_age["trch_age"].value_counts()
modalites_age = effectifs_age.index # l'index de effectifs contient les modalités

tab_age = pd.DataFrame(modalites_age, columns = ["trch_age"]) # création du tableau à partir des modalités
tab_age["n"] = effectifs_age.values
tab_age["f"] = tab_age["n"] / len(tranch_age) # len(data) renvoie la taille de l'échantillon

display(tab_age)


# In[1012]:


#frequences cumulées
tab_age = tab_age.sort_values("trch_age") # tri des valeurs de la variable X (croissant)
tab_age["F"] = tab_age["f"].cumsum() # cumsum calcule la somme cumulée

display(tab_age)
#revoir les traches dage


# # des indicateurs de tendance centrale et de dispersion (age)

# In[1013]:


trans_cust.describe()


# In[1014]:


#une mesure de tendance centrale 
trans_cust['age'].mode()


# In[1015]:


#la mediane du prix ds produits
trans_cust['age'].median()


# In[1016]:


round(trans_cust['age'].mean(),2)


# # les mesures de dispersion(age)

# In[1017]:



trans_cust.boxplot(column="age", vert=False)
plt.savefig("p4_figures/mesr_disp_age.png")
plt.show()


# # les mesures de forme

# In[1018]:


#calcul du skewness
trans_cust['age'].skew()

Si  γ1>0 alors la distribution est étalée à droite.
# In[1019]:


#calcul du kurtosis
trans_cust['age'].kurtosis()

on a  γ2>0  alors la distribution est moins aplatie que la distribution normale : les observations sont plus concentrées.
# In[ ]:





# # Analyser les ventes par tranche d'age

# ## Analyser l'age des clients eyants fait des transactions

# In[1020]:



trans_cust.age.min()


# In[1021]:


#on unifie les 3 tables trans_prod avec custumers 
trans_cust_prod=pd.merge(tranch_age,products)
trans_cust_prod.head(3)


# In[1022]:


trans_cust_prod.isnull().any()


# ### analyser la distribution emperique de la variable  age des clients 

# In[1023]:


trans_cust_prod.head


# In[1024]:


# Histogramme

# Diagramme en bâtons
trans_cust_prod["trch_age"].value_counts(normalize=True,sort=False).plot(kind='bar',width=0.5)
plt.savefig("p4_figures/evol_ventes_age.png")
plt.title('distribution des ventes entre les tranches d\'age')
plt.show()

ce diagrame se superpose sur le diagrame en baton des transaction, en fonction de la tranche d'age.
il demontre que la courbe s'acroit en partant des plus jeunes à l'adulte pour arriver à la tranche la plus dipenssive qui est entre  32 a 47 ans.
aprés la courbe décroit plus on est agé et mois on depensse  en age 
# # les mesures de concentration

# # analyser les ventes par raport a la nature des clients 

# ## Représentez la distribution empirique d'une variable

# ### 1. variables qualitatives

# In[1025]:


# Diagramme en secteurs
trans_cust_prod["sex"].value_counts(normalize=True).plot(kind='pie')
# Cette ligne assure que le pie chart est un cercle plutôt qu'une éllipse
plt.axis('equal') 
plt.savefig("p4_figures/dis_vente_sex_client_pie.png")
plt.title('distribution des ventes par raport au sex du client (camamber)')
plt.show() # Affiche le graphique

# Diagramme en tuyaux d'orgues
trans_cust_prod["sex"].value_counts(normalize=True).plot(kind='bar')
plt.title('distribution des ventes par raport au sex du client (bat)')
plt.savefig("p4_figures/dis_vente_sex_client_bat.png")
plt.show()

on remarque que la distribussion est presque équitable 
# ### le prix en fonction du sexe du clients

# In[1026]:


#boite à moustache

trans_cust_prod.boxplot(column='price', by='sex', showmeans=True)


plt.title('Dispersion des prix par sex')
plt.xlabel('sex')
plt.ylabel('prices')


plt.show()


# In[1027]:


#pour voir plus claire on travail avec un sous echantillon prix<70
#boite a moustache
echtrans_cust_prod=trans_cust_prod[trans_cust_prod.price<30]

sns.set_style("whitegrid")
sns.boxplot(data=echtrans_cust_prod,x="sex",y="price", palette="deep")
sns.despine(left=True)

#echtrans_cust_prod.boxplot(column='price', by='sex', showmeans=True)

plt.title('Dispersion des prix par sex')
plt.xlabel('sex')
plt.ylabel('prices')
plt.savefig("p4_figures/dis_prix_sex.png")

plt.show()

le boxplot montre une egalité des prix de vente dépenssé par les hommes et les femmes
# ## le prix en fonction de la categorie de produits

# In[1028]:


#boite à moustache

ech_trans_prod=trans_prod_net
#ech_trans_prod.boxplot(column='price', by='categ', showmeans=True)

sns.set_style("whitegrid")
sns.boxplot(data=ech_trans_prod,x="categ",y="price", palette="deep")


plt.title('Dispersion des prix par catégorie')
plt.xlabel('Catégories')
plt.ylabel('les prix des produits')
plt.ylim(0,100)
#plt.show()
plt.savefig("p4_figures/prix_categorie_produits.png")
sns.despine(left=True)

on remarque la différence des prix en fonction des categories 
la categorie 2 se demarque par des prix plus élevés 
# # Analyser deux variables qualitatives avec le Chi-2

# In[ ]:





# In[1029]:


X = "sex"
Y = "categ"

cont1 = trans_cust_prod[[X,Y]].pivot_table(index=X,columns=Y, aggfunc=len ,margins=True,margins_name="Total")
cont1


# In[1030]:


#mettre en % 


# In[1031]:


#Voici le code affichant cette heatmap 
import seaborn as sns

tx = cont1.loc[:,["Total"]]
ty = cont1.loc[["Total"],:]
n = len(trans_cust_prod)
indep = tx.dot(ty) / n
plt.figure(figsize=(12,8))
c = cont1.fillna(0) # On remplace les valeurs nulles par 0
measure = (c-indep)**2/indep
xi_n = measure.sum().sum()
table = measure/xi_n
sns.heatmap(table.iloc[:-1,:-1],annot=c.iloc[:-1,:-1])
plt.savefig("p4_figures/heat_map.png")
plt.title('Analysez deux variables qualitatives avec le Chi-2')
plt.show()


# In[1032]:


xi_n = measure.sum().sum()
xi_n 

le xi2 étant élvé on peut rejeter l hypothése null qui dit qu'il n'éxiste pas de relation entre le sex et categorie 
# In[1033]:


# Chi-square test of independence.
from scipy.stats import chi2_contingency
c, p, dof, expected = chi2_contingency(cont1)
p,dof

la p value étant tres loin de 5% on peu conclure à une relation entre sex et categorie 
# # Analyser la corrélation entre l'âge des clients (variable qualitative) et Le montant total des achats (variables quantitatives)
le diagramme de dispersion entre la tranche d'age et le mantant total des achats test anova 
tout les mean sont ==
au moin une des 4 est diferente des autres seuill de 5%

# In[1090]:


trans_cust_prod.head()


# In[1091]:


trans_cust_prod['tr_age']= pd.cut(trans_cust_prod['age'],5)


# In[1097]:


X = "trch_age" # qualitative
Y = "price" # quantitative(montant total des d'achat)
f, ax = plt.subplots(figsize=(10,5))

modalites_tr = trans_cust_prod[X].unique()
gp_tr = []
for m in modalites_tr:
    gp_tr.append(trans_cust_prod[trans_cust_prod[X]==m][Y])

# Propriétés graphiques (pas très importantes) 

medianprops= {'color':"black"}
meanprops = {'marker':'o', 'markeredgecolor':'black',
            'markerfacecolor':'firebrick'}
plt.title('La répartition des ventes par tranche d\'age' )
plt.xlabel('montant total des achats')
plt.ylabel('tranches d\'age')
    
plt.boxplot(gp_tr, labels=modalites_tr, showfliers=False, medianprops=medianprops, 
            vert=False, patch_artist=True, showmeans=True, meanprops=meanprops)
plt.savefig("p4_figures/rep_achat_age.png")
plt.show()

#On voit ici que les montants totaux des achas sont très différents de la tranche d'age de 17 à 30 ans au autre, ils sont plus élevés. Mais vérifions maintenant cette affirmation par les chiffres, grâce à une modélisation.On calcule 𝜂2  (eta carré ou eta squared, en anglais) :
# In[1099]:


X = "tr_age" # qualitative
Y = "price" # quantitative

def eta_squared_age(x,y):
    moyenne_age = y.mean()
    tranches = []
    for tranche in x.unique():
        yi_tranche = y[x==tranche]
        tranches.append({'ni': len(yi_tranche),
                        'moyenne_tranche': yi_tranche.mean()})
    SCT = sum([(yj-moyenne_age)**2 for yj in y])
    SCE = sum([c['ni']*(c['moyenne_tranche']-moyenne_age)**2 for c in tranches])
    return SCE/SCT
    
print(round(eta_squared_age(trans_cust_prod[X],trans_cust_prod[Y]),2))


# In[1038]:


#le rapport de corrélation etant superieur à 0 on laisse pensser qu il existe un rapport de corrélation entre l'age et le montant des achats mais qui exclus deux classes 


# # analyser par régression linéaire¶ 
on regroupe les ventes par age 
on prend un echantillion inferieur a 100 e
# In[1102]:


trans_cust_age =trans_cust_prod
trans_cust_age.head(5)


# In[1103]:


trans_cust_age = trans_cust_prod.groupby(["age"])['price'].agg('sum').reset_index()
trans_cust_age.head(3)


# In[1104]:


trans_cust_prod_age= trans_cust_age[['age', 'price']].sort_values(by='age', ascending=True)
trans_cust_prod_age.head(3)

Essayons d'afficher le diagramme de dispersion avec X = age et Y = price, et regardons si tous les points sont alignés :
# In[1105]:


plt.plot(trans_cust_prod_age['age'],trans_cust_prod_age['price'],'+', color='green')

plt.xlabel("age")
plt.ylabel("le prix des achats")
plt.title('Montant Total des achats en fonction de l\'âge des clients')
plt.savefig("p4_figures/mont_achat.png")
plt.show()

effectuer une régression linéaire et estimation de a et b
# In[1106]:


import statsmodels.api as sm

Y = trans_cust_prod_age['price']
X = trans_cust_prod_age[['age']]
X = X.copy() # On modifiera X, on en crée donc une copie
X['intercept'] = 1.
result = sm.OLS(Y, X).fit() # OLS = Ordinary Least Square (Moindres Carrés Ordinaire)
a,b = result.params['age'],result.params['intercept']
result.pvalues

p value valeur ascocie a a % de confience par raport ou a = ou !de 0
la pente est proche de 0 notre pente est different e de 0 avec un risque <5%
donc les variables sont liées 
dans un sense - 
 tracer la droite de régression à partir des coefficients obtenus :
# In[1107]:


plt.plot(trans_cust_prod_age.age,trans_cust_prod_age.price, "+")
plt.plot(np.arange(100),[a*x+b for x in np.arange(100)])
plt.xlabel("age")
plt.ylabel("montant des achats")
plt.title('Montant Total des achats en fonction de l\'âge des clients (droite de regression lineaire)')
plt.savefig("p4_figures/evolution_ventes.png")
plt.show()

#on peut remarquer deux types de comportement par tranche d'age donc on peut penser à decouper en deux tranches de 18 a 40 ans et de 40 a 97ans 
# ### calculer le coefficient de corrélation linéaire de Pearson
# 
pour avoir un resultat plus fiable on travail avec des données non biaisées donc avec la table trans_cust_prod
# In[1108]:


import scipy.stats as st
import numpy as np

print(st.pearsonr(trans_cust_prod["age"],trans_cust_prod["price"])[0])
print(np.cov(trans_cust_prod["age"],trans_cust_prod["price"],ddof=0)[1,0])

#le coefficient de corrélation linéaire de Pearson se situ entre -1 a 1 donc on peut dire que les deux variables sont fortement corellées et vue que -0.2 est negatif, on peut dire qu'a partir de l age de 40 ans plus les les clients sont agées et moins sont leurs dépenses 
# In[1109]:


#on essay de découper en deux la 


# # Y a-t-il une corrélation entre l'âge des clients et La fréquence d’achat
#  ( pour chaque client (age des clients len nombre d'achats totaux sur l enssemble du temps des achats pour chaque mois );

# In[1110]:


#on group par mois les achats et on calcul le len


# In[1111]:


gb_trans_cust_len = trans_cust_prod.groupby([trans_cust_prod['date'].dt.strftime('%Y-%m'),"client_id","age"])['session_id'].agg('count').reset_index()
gb_trans_cust_len.head()


# In[1112]:


ord_trans_cust_len=gb_trans_cust_len.sort_values('session_id', ascending = False)

ord_trans_cust_len.head()


# In[1113]:


gb_trans_cust_mois = ord_trans_cust_len.groupby(['date']).agg('count').reset_index()
gb_trans_cust_mois.head()


# In[1114]:


gb_trans_cust_merg=pd.merge(gb_trans_cust_mois,ord_trans_cust_len, on='date', how='left')
gb_trans_cust_merg.head()


# In[1115]:


#calculer la frequence d'achats mensuels
gb_trans_cust_merg['fréquence']=(gb_trans_cust_merg.session_id_y/gb_trans_cust_merg.session_id_x).apply(lambda x: round(x,3))
gb_trans_cust_merg.head()


# In[1116]:


gb_trans_cust_merg.isnull().sum()


# In[1117]:


gb_trans_cust_tri=gb_trans_cust_merg.sort_values(by=['date','age_y'], ascending=True)
gb_trans_cust_tri.head()


# In[1118]:


trans_cust_prod[trans_cust_prod.client_id=='c_1609'].sort_values(by='date').reset_index().head()


# In[1119]:


#il n ya pas de probleme avec le client


# In[1120]:



#Visualisation avec un scatterplot 
plt.plot(gb_trans_cust_tri.age_y,gb_trans_cust_tri.session_id_y, '+')

plt.xlabel('age')
plt.ylabel('Fréquence')
plt.title('Fréquence des achats selon l\'age du client')
plt.ylim(0, 100)
plt.savefig("p4_figures/fr_ventes.png")
plt.show()


# In[1121]:


#mettre l age en classe qualitatif comme dans age et mantant des achats


# In[1122]:


import statsmodels.api as sm

Y = gb_trans_cust_tri['session_id_y']
X = gb_trans_cust_tri[['age_y']]
X = X.copy() # On modifiera X, on en crée donc une copie
X['intercept'] = 1.
result = sm.OLS(Y, X).fit() # OLS = Ordinary Least Square (Moindres Carrés Ordinaire)
a,b = result.params['age_y'],result.params['intercept']


# In[ ]:





# In[1123]:


plt.plot(gb_trans_cust_tri.age_y,gb_trans_cust_tri.session_id_y, "+")
plt.plot(np.arange(100),[a*x+b for x in np.arange(100)])
plt.xlabel("age_y")
plt.ylabel("session_id_y")
plt.ylim(0, 50)
plt.title('Fréquence des achats selon l\'age du client (droite de régression lineaire)')
plt.savefig("p4_figures/fr_ventes_droite.png")
plt.show()


# In[1124]:


#pour une eaugmentation de 10 ans de nos client notre frequence d'acaht diminue de 0.17 articles 
# In[1125]:


a*40+b

mettre l axe y entre 0 et 100 
# In[1063]:


#Coefficient de corrélation linéaire de Pearson
print(st.pearsonr(gb_trans_cust_tri["age_y"],gb_trans_cust_tri["session_id_y"])[0])
print(np.cov(gb_trans_cust_tri["age_y"],gb_trans_cust_tri["session_id_y"],ddof=0)[1,0])

il ya une corelation faible 
# ## Analyser avec ANOVA 

# In[1064]:


gb_trans_cust_tri


# In[1127]:


gb_trans_cust_tri['trch_age_y'] = pd.cut(gb_trans_cust_tri['age_y'],5)


# In[1128]:


gb_trans_cust_tri.sort_values(by='trch_age_y',ascending=True)


# In[1131]:


X = "trch_age_y" # qualitative
Y = "session_id_y" # quantitative(fréquence d'achat)

f, ax = plt.subplots(figsize=(10,5))

modalites_fr = gb_trans_cust_tri[X].unique()
gp_fr = []
for m in modalites_fr:
    gp_fr.append(gb_trans_cust_tri[gb_trans_cust_tri[X]==m][Y])

# Propriétés graphiques 

medianprops_fr= {'color':"black"}
meanprops_fr = {'marker':'o', 'markeredgecolor':'black',
            'markerfacecolor':'firebrick'}
plt.title('La répartition des fréquences d\'achat par tranche d\'age' )
plt.xlabel('montant total des achats')
plt.ylabel('tranches d\'age')
    
plt.boxplot(gp_fr, labels=modalites_tr, showfliers=False, medianprops=medianprops_fr, 
            vert=False, patch_artist=True, showmeans=True, meanprops=meanprops_fr)
plt.savefig("p4_figures/repart_frequence_achat_age.png")
plt.show()


# In[ ]:




la moyenne pour 32 ans est tres differente des autres 'le calcule à la main  𝜂2  (eta carré ou eta squared, en anglais). le calcul  ;) :

# In[1132]:



X = "trch_age_y" # qualitative
Y = "session_id_y" # quantitative

def eta_squared(x,y):
    moyenne_y = y.mean()
    classes = []
    for classe in x.unique():
        yi_classe = y[x==classe]
        classes.append({'ni': len(yi_classe),
                        'moyenne_classe': yi_classe.mean()})
    SCT = sum([(yj-moyenne_y)**2 for yj in y])
    SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes])
    return SCE/SCT
    
eta_squared(gb_trans_cust_tri[X],gb_trans_cust_tri[Y])

#en regéte la vriable h0 
#la distribution est en génerale normale en comparaison avec une moyenne noramale avec un écart type, 
#la distribution s'écart et se raproche de 0 
# In[ ]:





# # Y a-t-il une corrélation entre l'âge des clients et Les catégories de produits achetés:
# 
#     

# ## Analysez une variable quantitative et une qualitative par ANOVA entre la categorie des produits et l'age des clients

# In[1073]:


X = "categ" # qualitative
Y = "age" # quantitative
 


# In[1074]:


modalites_cat = trans_cust_prod[X].unique()
groupes_cat = []
for m in modalites_cat:
    groupes_cat.append(trans_cust_prod[trans_cust_prod[X]==m][Y])

# Propriétés graphiques (pas très importantes)    
medianprops = {'color':"black"}
meanprops = {'marker':'o', 'markeredgecolor':'black',
            'markerfacecolor':'firebrick'}
plt.title('La répartition de l\'age selon les categories' )
plt.xlabel('age')
plt.ylabel('categories')
    
plt.boxplot(groupes_cat, labels=modalites_cat, showfliers=False, medianprops=medianprops, 
            vert=False, patch_artist=True, showmeans=True, meanprops=meanprops)
plt.savefig("p4_figures/repartition_age_categ.png")
plt.show()

on remarque d'apres les boites a moustaches une relation forte entre l age et les categiries tel que la categorie 1 est demandé par presque tout les clients avec une distribition presque homogéne mais qui plais surtout au populations de 40 a 60 ans pendant que la categorie 2 se limite a une clientelle tres jeune
et apres avoir fait l'analyse des pris des produit ainssi que la repartition des vente selon l ae on peut conclure que la categorie 2 comprent des prix chers ce qui fait qu'elle soit atractive pour les plus jeune et ce qui explique aussi que les jeunnes depencent plus que les autres tranches bien qu'ils ne font pas plus de trnsaction
# In[1075]:


X = "categ" # qualitative
Y = "age" # quantitative

def eta_squared_cat(x,y):
    moyenne_cat = y.mean()
    classes_cat = []
    for classe_cat in x.unique():
        yi_classe_cat = y[x==classe_cat]
        classes_cat.append({'ni': len(yi_classe_cat),
                        'moyenne_classe_cat': yi_classe_cat.mean()})
    SCT = sum([(yj-moyenne_cat)**2 for yj in y])
    SCE = sum([c['ni']*(c['moyenne_classe_cat']-moyenne_cat)**2 for c in classes_cat])
    return SCE/SCT
    
eta_squared_cat(trans_cust_prod[X],trans_cust_prod[Y])

le rapport de corrélation etant superieur a 0 On obtient un résultat proche de 0.1, ce qui laisse penser qu'il y a effectivement une corrélation entre l'age et les catégories
# In[1076]:


trans_cust_prod.head()


# # Y a-t-il une corrélation entre l'âge des clients et La taille du panier moyen (en nombre d’articles)

# ## Analysez deux variables quantitatives par régression linéaire

# In[1077]:


age_nb_art=trans_cust_prod.groupby(["client_id","age"])['session_id'].agg('count').reset_index()
age_nb_art


# In[1078]:


age_nb_pan=age_nb_art.rename(columns={'session_id':'panier'})

#afficher le diagramme de dispersion, on essay d enlever les outlier, On peut à présent effectuer notre régression linéaire.
#Voici comment estimer a et b avec Python. 
#créer les variables a et b contenant les estimations, je trace la droite de régression à partir des coefficients obtenus :
# ## Analysez la moyenne du panier deux variables quantitatives par régression linéaire

# In[1079]:


age_nb_mean=age_nb_pan.groupby(["age"])['panier'].agg('mean').reset_index().apply(lambda x: round(x, 2))
age_nb_mean.head()


# In[ ]:




afficher le diagramme de dispersion
# In[1080]:


plt.plot(age_nb_mean['age'],age_nb_mean['panier'], "+")
plt.xlabel("age")
plt.ylabel("panier")
plt.title('distribussion du panier moyen des client selon l age selon l\'age du client')
plt.savefig("p4_figures/rep_age_pan.png")
plt.show()

On peut à présent effectuer notre régression linéaire, créer les variables a et b contenant les estimations
# In[1081]:


import statsmodels.api as sm

Y3 = age_nb_mean['panier']
X3 = age_nb_mean[['age']]
X3 = X3.copy() # On modifiera X, on en crée donc une copie
X3['intercept'] = 1.
result3 = sm.OLS(Y3, X3).fit() # OLS = Ordinary Least Square (Moindres Carrés Ordinaire)
a3,b3 = result3.params['age'],result3.params['intercept']


# In[1082]:


print(a3,b3)


# In[1137]:


plt.plot(age_nb_mean.age,age_nb_mean.panier, "*")
plt.plot(np.arange(100),[a3*x+b3 for x in np.arange(100)])
plt.xlabel("age")
plt.ylabel("panier")
plt.title('distribussion du panier moyen en fonction de l\'age avec la courbe de regression lineaire ')
plt.savefig("p4_figures/rep_panier_age2.png")
plt.ylim(0,70)
plt.show()


# j enleve les outlier 

# In[1138]:


ech_age_nb_mean = age_nb_mean

Y3 = ech_age_nb_mean['panier']
X3 = ech_age_nb_mean[['age']]
X3 = X3.copy() # On modifiera X, on en crée donc une copie
X3['intercept'] = 1.
result3 = sm.OLS(Y3, X3).fit() # OLS = Ordinary Least Square (Moindres Carrés Ordinaire)
a3_new,b3_new = result3.params['age'],result3.params['intercept']

print(result3.params)

plt.plot(ech_age_nb_mean.age,ech_age_nb_mean.panier, "*")
plt.plot(np.arange(100),[a3_new*x+b3_new for x in np.arange(100)])
plt.plot(np.arange(100),[a3*x+b3 for x in np.arange(100)])
plt.xlabel("age")
plt.ylabel("panier")
plt.savefig("p4_figures/repartition_pan_age-clients.png")
plt.title('distribussion du panier selon l\'age des clients')
plt.ylim(0,80)
plt.show()


# In[1085]:


#d apres le graphe on remarque qu'il ya une certaine disparité en fonction des tranches d'age qui se devise en trois de 17 a 30 ans on remarque que le nombre de panier moyen est tres faible contrairement à la tranche adult de 30 a 50 qui se distingue par un panier moyen tres elevé pour un panier moyen de 50 a 97 ans qui prend plus de poid  


# In[1086]:


a3_new,b3_new


# ### calculer le coefficient de Pearson et la covariance,

# In[1087]:


print(st.pearsonr(age_nb_mean["age"],age_nb_mean["panier"])[0])
print(np.cov(age_nb_mean["age"],age_nb_mean["panier"],ddof=0)[1,0])

le coefficient de corrélation linéaire de Pearson se situ entre -1 a 1 donc on peut dire que les
deux variables sont fortement corellées et vue que -0.26 est plus proche tand vers -1 donc on peut
dire que a partir de 50ans  plus les les clients sont agées et moins sont leurs paniers moyen est
important 





