package org.vanish;

import org.apache.log4j.BasicConfigurator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class IrisClassification {
    private static final int FEATURES_COUNT = 4;
    private static final int CLASSES_COUNT = 3;

    public static void main(String[] args) {
        BasicConfigurator.configure();
        loadData();

    }

    private static void loadData() {
        try(RecordReader recordReader = new CSVRecordReader(0,',')) {
            recordReader.initialize(new FileSplit(
                    new ClassPathResource("iris.csv").getFile()
            ));

            //we’ll iterate over the dataset
            DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, 150, FEATURES_COUNT, CLASSES_COUNT);
            DataSet allData = iterator.next();
            allData.shuffle(0);

            //we’ll normalize the data (fit-transform)
            DataNormalization normalizer = new NormalizerStandardize();
            normalizer.fit(allData);
            normalizer.transform(allData);

            //we split the data
            SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65);
            DataSet trainingData = testAndTrain.getTrain();
            DataSet testingData = testAndTrain.getTest();


            //Running training and testing
            System.out.println("\n\n\nInitiating Deep Learning");
            irisNNetwork(trainingData, testingData);


        } catch (Exception e) {
            Thread.dumpStack();
            new Exception("Stack trace").printStackTrace();
            System.out.println("Error: " + e.getLocalizedMessage());
        }
    }

    private static void irisNNetwork(DataSet trainingData, DataSet testData) {
        System.out.println("\n\n\nModel Configuration");
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
//                .seed(0)
                .iterations(100)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(0.1, 0.9))
                .l2(0.0001)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(FEATURES_COUNT).nOut(2).build())
                .layer(1, new DenseLayer.Builder().nIn(2).nOut(2).build())
                .layer(2, new OutputLayer.Builder(
                        LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX)
                        .nIn(2).nOut(CLASSES_COUNT).build())
                .backprop(true).pretrain(false)
                .build();

        System.out.println("\n\n\nModel Training");
        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.setListeners(new PerformanceListener(1,true));
        model.init();
        model.fit(trainingData);

        //Evaluating over training data
        System.out.println("\n\nEvaluating over training data");
        INDArray output2 = model.output(trainingData.getFeatureMatrix());
        Evaluation eval2 = new Evaluation(CLASSES_COUNT);
        eval2.eval(trainingData.getLabels(), output2);
        System.out.printf(eval2.stats());

        //Evaluating over test data
        System.out.println("\n\nEvaluating over test data");
        INDArray output = model.output(testData.getFeatureMatrix());
        Evaluation eval = new Evaluation(CLASSES_COUNT);
        eval.eval(testData.getLabels(), output);
        System.out.printf(eval.stats());

    }
}
