#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
plt.rcParams['figure.figsize'] = (15, 5)


# In[2]:


df= pd.read_csv(r"C:\Users\isa_m\Dropbox\Prueba Big holding\ML\prueba_clientes_bh.csv")
df.head()


# In[3]:


#Renombrar las columnas del df
df.columns=["usuario_afiliado", "recencia", "frecuencia", "ticket_promedio", "total_ventas"]
df.head()


# In[4]:


df.info()


# In[5]:


stats= df.describe()
stats


# In[6]:



plt.subplot(1,2,1)
df.boxplot(column= 'recencia')
plt.title('Recencia')
plt.show()

plt.subplot(1,2,2)
df.boxplot(column= 'frecuencia')
plt.title('Frecuencia')
plt.show()


# In[7]:


plt.subplot(1,2,1)
df.boxplot(column= 'ticket_promedio')
plt.title('ticket_promedio')
plt.show()

plt.subplot(1,2,2)
df.boxplot(column= 'total_ventas')
plt.title('total_ventas')
plt.show()


# In[ ]:





# In[8]:


#Teniendo en cuenta que el negocio es de retail, se decide retirar del DS cualquier precio por debajo de los 400 pesos.
#Fueron eliminadas 67 filas
df.drop(range(236546, 236612, 1), axis=0, inplace= True)
df.tail()


# In[9]:


df.drop(236612, axis=0, inplace=True)
df.tail()


# In[10]:


stats= df.describe()
stats


# In[11]:


df.head()


# # K-means 

# Supuestos:
# 
# -Las variables están simétricamente distribuidas (no hay sesgos)
# -Las variables tienen los mismo valores promedio
# -Las variables tienen la misma varianza

# In[12]:


#Buscar variables con sesgos:


# In[13]:


sns.distplot(df['recencia'])
plt.show()


# In[14]:


sns.distplot(df['frecuencia'])
plt.show()


# In[15]:


sns.distplot(df['ticket_promedio'])
plt.show()


# In[16]:


sns.distplot(df['total_ventas'])
plt.show()


# In[17]:


import numpy as np


# # Transformación de variables:

# In[18]:


df.info()


# In[ ]:





# In[19]:


df['recencia']= df['recencia']+0.0001

rec_log= np.log(df['recencia'])
sns.distplot(df['recencia'])
plt.show()


# In[20]:


freq_log= np.log(df['frecuencia'])
sns.distplot(freq_log)
plt.show()


# In[21]:


ticket_log= np.log(df['ticket_promedio'])
sns.distplot(ticket_log)
plt.show()


# In[22]:


ventas_log= np.log(df['total_ventas'])
sns.distplot(ventas_log)
plt.show()


# In[23]:


sns.scatterplot(x= 'recencia', y='total_ventas', data= df)


# In[ ]:





# In[24]:


sns.scatterplot(x= 'frecuencia', y='total_ventas', data= df)


# In[ ]:





# In[25]:


corr=df.corr()


# In[26]:


sns.heatmap(corr, 
           xticklabels=corr.columns,
           yticklabels=corr.columns, annot=True)


# In[ ]:





# In[ ]:





# # Centrado y escalado de variables

# In[27]:


df.drop(['usuario_afiliado','ticket_promedio'], axis=1, inplace=True)
df


# # Combinar escalado y centrado:

# In[28]:


from sklearn.preprocessing import StandardScaler


# In[29]:


scaler=StandardScaler()
scaler.fit(df)
df_normalized= scaler.transform(df)


# In[30]:


print('mean: ', df_normalized.mean(axis=0).round(2))
print('std: ', df_normalized.std(axis=0).round(2))


# # Entrenar modelo. Definir número de Clusters

# In[31]:


from sklearn.cluster import KMeans


# In[32]:


#Iniciar modelo
model= KMeans(n_clusters=2, random_state=1)


# In[33]:


#Entrenar modelo
model.fit(df_normalized)

cluster_labels= model.labels_


# In[34]:


df_k2= df.assign(Cluster= cluster_labels)


# In[35]:


df_k2.groupby(['Cluster']).agg({
    'recencia': 'mean',
    'frecuencia': 'mean',
    'total_ventas': ['mean', 'count'],
}).round(2)


# In[36]:


#Ajustar Kmeans y calcular SSE para cada K

sse= {}



# In[37]:


for k in range(1,11):
    model= KMeans(n_clusters =k, random_state=1)
    model.fit(df_normalized)
    sse[k]= model.inertia_
   


# In[38]:


#Graficar SSE para cada K

plt.title('Metodo Elbow')
plt.xlabel('k'); plt.ylabel('SSE')
sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
plt.show()


# In[39]:


#Modelo con 3 clusters de acuerdo al Elbow Criterion
model_3= KMeans(n_clusters=3, random_state=1)


# In[40]:


#Entrenar modelo
model_3.fit(df_normalized)

cluster_labels_3= model_3.labels_

df_k3= df.assign(Cluster= cluster_labels_3)

df_k3.groupby(['Cluster']).agg({
    'recencia': 'mean',
    'frecuencia': 'mean',
    'total_ventas': ['mean', 'count'],
}).round(2)


# In[41]:


df_k3['label']= cluster_labels_3


# In[ ]:





# El cluster 2 es el que tiene menor recencia media, mayor frecuencia y mayor total de ventas en promedio. 
# Se asigna al Grupo Alto.
# 
# El Cluster 0 se asigna al grupo Medio. Es el siguiente con menor recencia y mayor frecuencia de compra.
# 
# Por último, el cluster 1 se asigna al grupo bajo. Son los clientes con mayor recencia y menor frecuencia de compra
# 

# In[42]:


from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'inline')
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')


# In[43]:


df_k3.head()


# In[44]:


cluster_labels_3= model_3.labels_
C = model_3.cluster_centers_



colores=['red','green','cyan']
asignar=[]
for row in cluster_labels_3:
    asignar.append(colores[row])
 
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(df_k3['recencia'], df_k3['frecuencia'], df_k3['total_ventas'], c=asignar,s=60)
#ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c=colores, s=1000)


# In[ ]:





# In[ ]:




