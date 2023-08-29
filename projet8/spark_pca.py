from pyspark.ml.feature import VectorSlicer
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import Row
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.feature import PCAModel, PCA, MinMaxScaler, StringIndexer, VectorIndexer
from pyspark.ml.feature import IndexToString

from pyspark.sql.functions import col, udf, pandas_udf, PandasUDFType, element_at, split
from pyspark.ml.functions import vector_to_array, array_to_vector
from pyspark.ml import Pipeline, Transformer

from pyspark import keyword_only
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params, TypeConverters
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.types import StringType


import os
import io
import time
from socket import gethostname

import numpy as np
import pandas as pd
from PIL import Image

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

is_aws = gethostname()!='nordine-optiplex7040'

if not is_aws:
    PATH = os.getcwd()
    PATH_Result = PATH + '/Results_PCA2'
    PATH_pipe_model = PATH + '/pipeline_model'
    PATH_input = PATH + '/data/Test'
else:
    PATH = '/kockot-bucket'
    PATH_Result = 's3://kockot-bucket/Results_PCA'
    PATH_pipe_model = 's3://kockot-bucket/pipeline_model'
    PATH_input = 's3://kockot-bucket/Test'


if not os.path.exists(PATH_Result):
    os.mkdir(PATH_Result)
    
t0 = time.time()

if not is_aws:
    spark = (SparkSession
             .builder
             .appName('P8')
             .config("spark.sql.parquet.writeLegacyFormat", 'true')
             .getOrCreate()
    )
else:
    spark = (SparkSession
             .builder
             .appName('P8_aws')
             .config("spark.sql.parquet.writeLegacyFormat", 'true')
             .getOrCreate()
    )

sc = spark.sparkContext

class CustomTransformer(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable):

   @keyword_only
   def __init__(self, inputCol=None, outputCol=None):
       super(CustomTransformer, self).__init__()
       kwargs = self._input_kwargs
       self.setParams(**kwargs)

   @keyword_only
   def setParams(self, inputCol=None, outputCol=None):
       kwargs = self._input_kwargs
       return self._set(**kwargs)

   def _transform(self, dataset):
       r= dataset.select(
           col("path"), 
           col("label"),
           array_to_vector(self.getInputCol()).alias(self.getOutputCol())
        )
       return r


def model_fn(weights=None):
    """
    Returns a MobileNetV2 model with top layer removed 
    and broadcasted pretrained weights.
    """
    model = MobileNetV2(weights='imagenet',
                        include_top=True,
                        input_shape=(224, 224, 3))
    for layer in model.layers:
        layer.trainable = False
    new_model = Model(inputs=model.input,
                  outputs=model.layers[-2].output)
    #broadcast_weights = sc.broadcast(new_model.get_weights())
    if weights is not None:
        new_model.set_weights(weights)
    return new_model

new_model = model_fn()
broadcast_weights = sc.broadcast(new_model.get_weights())
broadcast_model_json = sc.broadcast(new_model.to_json())

class ImageDataSetTransformer(Transformer, DefaultParamsWritable, DefaultParamsReadable):

   @keyword_only
   def __init__(self):
       super(ImageDataSetTransformer, self).__init__()
       kwargs = self._input_kwargs
       self.setParams(**kwargs)

   @keyword_only
   def setParams(self):
       kwargs = self._input_kwargs
       return self._set(**kwargs)

   def _transform(self, dataset):
       return dataset.withColumn("label", element_at(split(dataset["path"], "/"), -2))


def preprocess(content):
    """
    Preprocesses raw image bytes for prediction.
    """
    img = Image.open(io.BytesIO(content)).resize([224, 224])
    arr = img_to_array(img)
    return preprocess_input(arr)

def featurize_series(model, content_series):
    """
    Featurize a pd.Series of raw images using the input model.
    :return: a pd.Series of image features
    """
    with tf.device("/CPU:0"):
        input = np.stack(content_series.map(preprocess))
        preds = model.predict(input)
        # For some layers, output features will be multi-dimensional tensors.
        # We flatten the feature tensors to vectors for easier storage in Spark DataFrames.
        output = [p.flatten() for p in preds]
        return pd.Series(output)

@pandas_udf('array<float>', PandasUDFType.SCALAR_ITER)
def featurize_udf(content_series_iter):
    '''
    This method is a Scalar Iterator pandas UDF wrapping our featurization function.
    The decorator specifies that this returns a Spark DataFrame column of type ArrayType(FloatType).

    :param content_series_iter: This argument is an iterator over batches of data, where each batch
                              is a pandas Series of image data.
    '''
    
    # With Scalar Iterator pandas UDFs, we can load the model once and then re-use it
    # for multiple data batches.  This amortizes the overhead of loading big models.
    model = tf.keras.models.model_from_json(broadcast_model_json.value)
    model.set_weights(broadcast_weights.value)
    for content_series in content_series_iter:
        yield featurize_series(model, content_series)


class FeatureTransformer(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable):

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, model=None):
        super(FeatureTransformer, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        customf = udf(lambda x: array_to_vector(x))(dataset[self.getInputCol()])
        r= dataset.select(
            col("path"), 
            col("label"),
            featurize_udf(self.getInputCol()).alias(self.getOutputCol())
        )
        return r


idst = ImageDataSetTransformer()
ft = FeatureTransformer(inputCol="content", outputCol="features")
arrayToVector = CustomTransformer(inputCol="features", outputCol="features2")
stringIndexer = StringIndexer(inputCol="label", outputCol="label_index")
minMaxScaler = MinMaxScaler(inputCol="features2", outputCol="scaled")
#pca = PCA(k=1024, inputCol="scaled", outputCol="pca")
pca = PCA(k=1024, inputCol="features2", outputCol="pca")

#pipe = Pipeline(stages=[idst, ft, arrayToVector, stringIndexer, minMaxScaler, pca])
pipe = Pipeline(stages=[idst, ft, arrayToVector, stringIndexer, pca])

images = spark.read.format("binaryFile") \
  .option("pathGlobFilter", "*.jpg") \
  .option("recursiveFileLookup", "true") \
  .load(PATH_input)
  
pipe_model = pipe.fit(images)

df = pipe_model.transform(images)

df.write.mode("overwrite").parquet(PATH_Result)
pipe_model.save(PATH_pipe_model)

print(f"durée d'execution: {time.time() - t0}")

w = True
while w:
    a = input("Appuyez sur Entrée pour arrêter: ")
    if a == '':
        w=False


