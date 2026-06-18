# Jupyter + PySpark 3.5.8 + Iceberg 1.11.0, JDK 17 (Iceberg 1.11 baseline).
# No prebuilt jupyter/pyspark-notebook ships Spark 3.5.x anymore (quay keeps only
# the latest, now Spark 4.x) -> we roll our own. Iceberg jars are BAKED into
# $SPARK_HOME/jars at build time: no runtime Ivy resolution, fully offline/reproducible.
FROM python:3.11-slim-bookworm

ARG SPARK_VERSION=3.5.8
ARG ICEBERG_VERSION=1.11.0
ARG SCALA=2.12
# Hadoop S3A (for Hadoop-FS ops like remove_orphan_files). MUST match Spark's bundled
# Hadoop (3.5.x -> 3.3.4); aws-java-sdk-bundle is the version hadoop-aws:3.3.4 depends on.
ARG HADOOP_VERSION=3.3.4
ARG AWS_SDK_VERSION=1.12.262

RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless curl procps && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        pyspark==${SPARK_VERSION} \
        jupyterlab==4.3.0 \
        pandas==2.2.3 \
        pyarrow==17.0.0

# Expose the pip-bundled Spark distribution at a stable path.
RUN ln -s "$(python -c 'import pyspark, os; print(os.path.dirname(pyspark.__file__))')" /opt/spark
ENV SPARK_HOME=/opt/spark
ENV PATH=$SPARK_HOME/bin:$PATH
RUN mkdir -p $SPARK_HOME/conf

# Bake Iceberg runtime + AWS bundle into the Spark classpath.
RUN M=https://repo1.maven.org/maven2/org/apache/iceberg && \
    curl -fsSLo $SPARK_HOME/jars/iceberg-spark-runtime-${SCALA}-${ICEBERG_VERSION}.jar \
      $M/iceberg-spark-runtime-3.5_${SCALA}/${ICEBERG_VERSION}/iceberg-spark-runtime-3.5_${SCALA}-${ICEBERG_VERSION}.jar && \
    curl -fsSLo $SPARK_HOME/jars/iceberg-aws-bundle-${ICEBERG_VERSION}.jar \
      $M/iceberg-aws-bundle/${ICEBERG_VERSION}/iceberg-aws-bundle-${ICEBERG_VERSION}.jar

# Hadoop S3A so Spark's Hadoop FileSystem layer understands s3:// (needed by remove_orphan_files).
RUN curl -fsSLo $SPARK_HOME/jars/hadoop-aws-${HADOOP_VERSION}.jar \
      https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/${HADOOP_VERSION}/hadoop-aws-${HADOOP_VERSION}.jar && \
    curl -fsSLo $SPARK_HOME/jars/aws-java-sdk-bundle-${AWS_SDK_VERSION}.jar \
      https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/${AWS_SDK_VERSION}/aws-java-sdk-bundle-${AWS_SDK_VERSION}.jar

RUN useradd -m -u 1000 jovyan
USER jovyan
WORKDIR /home/jovyan/work
EXPOSE 8888 4040
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", \
     "--ServerApp.token=aml", "--ServerApp.root_dir=/home/jovyan/work"]
