/*
David Shagam
W01027008
Program 2
Machine Learning CSCI 497E
 */


//General Java Imports
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.text.DecimalFormat;

//Apache imports
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

//Program to compute Binary logistic regression and MultiClass logistic regression.
//Also uses L2 normilization.
//Doesn't currently use line Backtracking
//Takes 4 to 9 arguments. optional arguments are in the form [-BT alpha b] and [-L2 lambda] these come before the
//4 required arguments which are file pahts which are the general config, the data config, the training data, and the dev data
public class Prog2 {

	public static void main(String[] args) {
        argParse(args);
	}

    //Takes an array of strings consisting of the arguments that will be parsed.
	public static void argParse(String[] args){
        //Check if we have sufficient args error if not.
		if(args.length <4 || args.length>9){
			System.out.println("Not enough or too many arguments Minimum number of arguments is 4 maximum is 9\n"
					+ "format is ./prog2 [-BT alpha b] [-L2 lambda] sys_cfg_fn data_cfg_fn train_fn dev_fn");
			System.exit(1);
		}
        //Setup default values
        int argNum = 0;
		String trainData = "";
		String devData = "";
		double alpha = -1;
		double b = -1;
		double lambda = 0;
		boolean backTrack = false;
		boolean l2Norm = false;
        //Check twice for the optional arguments They can be provided in any order hence the loop.
		for(int i=0; i < 2; i++){
            //Check for existence of flag
			if(args[argNum].equals("-BT")){
				backTrack = true;
				argNum++;
				alpha = Double.valueOf(args[argNum]);
				argNum++;
				b = Double.valueOf(args[argNum]);
				argNum++;
                //If we have BT then our argNum is 3 greater than when we started so it points to the next arg
			}else if(args[argNum].equals("-L2")){
				l2Norm = true;
				argNum++;
				lambda = Double.valueOf(args[argNum]);
				argNum++;
                //if we have L2 th en argNum is 2 greater than starting to point to next arg
			}
		}

        //Load general config
		sysConfigData sysData = readSysConfig(args[argNum], backTrack, l2Norm, alpha, b, lambda);
		argNum++;
        //load data config
		trainDevConfig trainDevInfo = readDataConfig(args[argNum]);
		argNum++;

		//Load Train Data
        matrixWrapper trainMatrixes = loadData(args[argNum], trainDevInfo.N_TRAIN, trainDevInfo.D, trainDevInfo.C);
		argNum++;
        //Load dev data
		matrixWrapper devMatrixes = loadData(args[argNum], trainDevInfo.N_DEV, trainDevInfo.D, trainDevInfo.C);
        //At this point all config info and all data should be loaded.

        //Make sure we have a valid number of classes in our data
         if (trainDevInfo.C>=2){
            //Binary or multi
            gradientDescent(trainMatrixes, devMatrixes, trainDevInfo, sysData, backTrack, l2Norm);
        } else {
            // Invalid class input.
            System.out.println("Number of Classes recieved is less than 2. This is not allowed");
            System.exit(1);
        }
	}

    //Computes the sigmoid of a matrix
    //Input a RealMatrix that we are referencing
    //Output a modified copy of the input Matrix with values in sigmoid form.
    //The original passed in matrix is not modified.
    public static RealMatrix sigmoid(RealMatrix augment){
        RealMatrix returnMatrix = augment.copy();
        //Rows
        for(int i = 0; i < returnMatrix.getRowDimension(); i++)
            //columns
            for(int j = 0; j < returnMatrix.getColumnDimension(); j++){
                //i=row, j=col
                //this is 1/(1+e^val(i,j))
                returnMatrix.setEntry(i,j, (1/(1+Math.exp(-returnMatrix.getEntry(i,j)))));
            }
        return returnMatrix;
    }


    //Computes hte softmax of a matrix
    //Input the RealMatrix that we are referencing
    //Output a RealMatrix that is a modified copy of Input matrix
    //Original is not modified.
    public static RealMatrix softmax(RealMatrix augment){
        RealMatrix returnMatrix = augment.copy();

        for(int i =0; i < returnMatrix.getRowDimension(); i++){
            double runningTotal = 0;
            //change every value to e^(val(i,j))
            //update runningTotal for the line.
            for(int j = 0; j < returnMatrix.getColumnDimension(); j++){
                double val = Math.exp(returnMatrix.getEntry(i,j));
                runningTotal += val;
                returnMatrix.setEntry(i,j, val);
            }
            //Divide every value per line by the running total so we get e^(val(i,j)) / rowTotal
            for(int j = 0; j < returnMatrix.getColumnDimension(); j++){
                returnMatrix.setEntry(i,j, ((returnMatrix.getEntry(i,j))/runningTotal));
            }
        }
        return returnMatrix;
    }


    //Calculates the accuracy for a multi class model.
    //Input our estimated value matrix, our actual value matrix, and a string builder to build a string
    //output a double representing the accuracy which is num correct/ num total
    public static double multiModelAccuracy(RealMatrix estimate, RealMatrix actual, StringBuilder str){
        double totalNum = 0;
        double correctNum = 0;
        //for every value
        for(int i = 0; i< estimate.getRowDimension(); i++){
            //defaults
            int theClass = -1;
            double bestChance = -1;
            for(int j = 0; j< estimate.getColumnDimension(); j++){
                //if our estimate value is better than our best value update our estimated class and it's chance.
                if(estimate.getEntry(i,j)> bestChance){
                    bestChance = estimate.getEntry(i,j);
                    theClass = j;
                }
            }

            //Deal with string builder adding that estimate to our output string.
            str.append(theClass);
            str.append(" ");
            //If our estimate is equal to the ACTUAL class increment correctNum;
            if(theClass == actual.getEntry(i,0)){
                correctNum += 1;
            }

            totalNum += 1;
        }
        //return the double accuracy.
        return correctNum/totalNum;
    }

    //Calculates the accuracy for a binary model
    //Input estimate matrix, actual class label matrix, String builder for ouptut.
    //Output a double representing accuracy.
    public static double modelAccuracy(RealMatrix estimate, RealMatrix actual, StringBuilder str){
        double totalNum = 0;
        double correctNum = 0;
        //For every data point in matrix
        for(int i = 0; i< estimate.getRowDimension(); i++){
            for(int j = 0; j< estimate.getColumnDimension(); j++){
                //If we are above are at or above a 50% chance pick class 1 check if it's equal to the actual.
                if(estimate.getEntry(i,j) >=0.5 && actual.getEntry(i,j)==1){
                    correctNum = correctNum + 1;
                    str.append("1 ");
                    //else if were below 50% pick class 0 and check if it's actual
                } else if(estimate.getEntry(i,j) < 0.5 && actual.getEntry(i,j)==0){
                    correctNum = correctNum + 1;
                    str.append("0 ");
                }
            }
            totalNum = totalNum +1;
        }
        return correctNum/totalNum;
    }


    //Calculates the gradient Descent for binary and multi class classification.
    //Input: the training data, the dev data, the data config, the sys config, and if we backtrack or regularize.
    //output: 1 line to error reporting iterations and accuracies, 2 outputs the standard out with the train and dev class predictions
    public static void gradientDescent(matrixWrapper train, matrixWrapper dev, trainDevConfig dataInfo, sysConfigData conditions, boolean backtrack, boolean regularize){
        //Class cutoff point is .5 for binary.

        //set up default information
        int numIters = 0;
        int numBad = 0;
        double bestDevAccuracy = 0;
        //used for binary and multi
        double[][] betaModel = null;
        //used only for multi
        double[][] kronecker = null;

        //check which case it is. Binary or Multi?
        if(dataInfo.C ==2) {
            //binary case need a D+1x1 matrix
            betaModel = new double[dataInfo.D + 1][1];
        } else {
            //Multi case need a D+1xC matrix
            betaModel = new double[dataInfo.D+1][dataInfo.C];
            //make kronecker which is an NxC matrix
            kronecker = new double[dataInfo.N_TRAIN][dataInfo.C];
            for(int i = 0; i < dataInfo.N_TRAIN; i++){
                for(int j = 0; j < dataInfo.C; j++){
                    //if column# == class# set it to 1 else 0.
                    if(j == train.y.getEntry(i,0)){
                        kronecker[i][j] = 1;
                    } else {
                        kronecker[i][j]=0;
                    }
                }
            }
        }

        //Init the betaModel to all 0's just in case.
        for (int i = 0; i < dataInfo.D + 1; i++) {
            for (int j = 0; j < betaModel[i].length; j++) {
                betaModel[i][j] = 0;
            }
        }
        //Create the matrix for the model and for kronecker if applicable
        RealMatrix modalMatrix = MatrixUtils.createRealMatrix(betaModel);
        RealMatrix kroneckerDelta = null;
        if(dataInfo.C>2) {
            kroneckerDelta = MatrixUtils.createRealMatrix(kronecker);
        }

        //Iterate, while our iterations are < max and our consecutive bad iters < max bad
        while (numIters < conditions.MAX_ITERS && numBad < conditions.MAX_BAD_COUNT) {
            //gradient descent.
            //X*B
            RealMatrix sigmoidThis = train.x.multiply(modalMatrix);

            //apply sigmoid or apply softmax;
            if(dataInfo.C==2) {
                //Sigmoid(X*B)
                sigmoidThis = sigmoid(sigmoidThis);
            } else {
                //Softmax(X*B)
                sigmoidThis = softmax(sigmoidThis);
            }

            //intermediate matrix
            RealMatrix val = null;
            //If binary subtract from  y. Else subtract from kronecker
            if(dataInfo.C==2){
                val = train.y.subtract( sigmoidThis);
            } else {
                val = kroneckerDelta.subtract( sigmoidThis);
            }

            //(y-sigmoid(X*B) OR (kronecker-softmax(X*B)
            //This is the gradient.
            val = (train.x.transpose()).multiply(val);

            //Create the L2 modified matrix.
            RealMatrix Regularize = modalMatrix.copy();
            //SET ROW 1 TO 0's
            for(int i = 0; i < Regularize.getColumnDimension(); i++){
                Regularize.setEntry(0,i, 0.0);
            }
            //apply lambda*2
            Regularize = Regularize.scalarMultiply((conditions.LAMBDA * 2));
            //Get intermediate matrix of new gradient - regularization
            val = val.subtract(Regularize);
            //Apply step size.
            modalMatrix = modalMatrix.add(val.scalarMultiply(conditions.STEP_SIZE));

            //make estimates
            RealMatrix trainEstimate = train.x.multiply(modalMatrix);
            RealMatrix devEstimate = dev.x.multiply(modalMatrix);

            //sigmoid vs softmax check for predictions
            if(dataInfo.C==2) {
                trainEstimate = sigmoid(trainEstimate);
                devEstimate = sigmoid(devEstimate);
            } else {
                trainEstimate = softmax(trainEstimate);
                devEstimate = softmax(devEstimate);
            }

            //Calculate accuracy.
            double trainAccuracy = 0;
            double devAccuracy = 0;

            //Multi VS Binary
            StringBuilder trainString = new StringBuilder("train ");
            StringBuilder devString = new StringBuilder("dev ");
            if(dataInfo.C == 2) {
                trainAccuracy = modelAccuracy(trainEstimate, train.y, trainString);
                devAccuracy = modelAccuracy(devEstimate, dev.y, devString);
            } else {
                trainAccuracy = multiModelAccuracy(trainEstimate, train.y, trainString);
                devAccuracy = multiModelAccuracy(devEstimate, dev.y, devString);
            }
            //increment iters
            numIters++;
            //Check if our accuracy is better or not.
            if(devAccuracy <= bestDevAccuracy){
                numBad++;
            } else {
                bestDevAccuracy = devAccuracy;
                numBad = 0;
            }
            //Report information.
            DecimalFormat df = new DecimalFormat("0.000");
            System.err.println("Iter "+ String.format("%04d", numIters) +": trainAcc="+ df.format(trainAccuracy) +" testAcc="+ df.format(devAccuracy));
            System.out.println(trainString.toString().trim());
            System.out.println(devString.toString().trim());

        }
    }

    //Loads Train and Dev data.
    //Input is the File path, the number of lines the file should contain, the number of tokens it should contain,
    //and hte number of classes it should contain.
    //Returns a matrixWrapper which holds 2 RealMatrixes contianing the data.
	public static matrixWrapper loadData(String filePath, int N, int Dim, int C){

        FileReader readFile = null;
        BufferedReader inputScanner = null;

        //try to open and get info from file.
        try{
            readFile = new FileReader(filePath);
            inputScanner = new BufferedReader(readFile);
        } catch (FileNotFoundException e){
            e.printStackTrace();
        }

        //set up default matrixes
        double[][] xVals = new double[N][Dim+1];
        double[][] yVals = new double[N][1];
        //Make sure it's X hat
        for(int i = 0; i < N; i++){
            xVals[i][0]=1;
        }
        //read lines one at a time.
        String line = null;
        int lineNum = 0;
        try {
            while((line=inputScanner.readLine())!=null){
                if(lineNum > N){
                    System.out.println("Expected a file with " + N + " Lines but recieved a file that is longer");
                    System.exit(1);
                }
                //tokenize string
                String [] tokens = line.trim().split("\\s+");
                //Get class label
                yVals[lineNum][0]= Integer.valueOf(tokens[0]);
                if(tokens.length-1 > Dim){
                    System.out.println("Expected a " + Dim + " Dimensional x vector got something larger");
                    System.exit(1);
                }
                //Grab the x values
                for(int j = 1; j < Dim+1; j++ ){
                    xVals[lineNum][j]= Double.valueOf(tokens[j]);
                }
                lineNum++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        //returns matrix wrapper with xvals and yvals.
        return new matrixWrapper(MatrixUtils.createRealMatrix(xVals), MatrixUtils.createRealMatrix(yVals));
	}
	
	//a function to load the data config files
    //Input: a file path
    //Output: a helper class that consists of the data.
	public static trainDevConfig readDataConfig(String filePath){
        FileReader readFile = null;
        BufferedReader inputScanner = null;
        try {
            readFile = new FileReader(filePath);
            inputScanner = new BufferedReader(readFile);

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        //Default data values
        int nTrain = -1;
        int nDev = -1;
        int dim = -1;
        int classNum =-1;
        String line = null;
        //Read line by line
        try {
            while((line=inputScanner.readLine())!=null){
                //tokenize
                String [] tokens = line.trim().split("\\s+");
                //If statement to set values
                if(tokens[0].equals("N_TRAIN")){
                    nTrain = Integer.valueOf(tokens[1]);
                } else if(tokens[0].equals("N_DEV")){
                    nDev = Integer.valueOf(tokens[1]);
                } else if(tokens[0].equals("D")){
                    dim = Integer.valueOf(tokens[1]);
                } else if(tokens[0].equals("C")){
                    classNum = Integer.valueOf(tokens[1]);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        //creates a trainDevConfig class and returns
        return new trainDevConfig(nTrain, nDev, dim, classNum);
		
		
	}
	
	//loads the general system config settings.
    //Takes in a file path, if Backtrakc is enabled, if L2 is enabled, the alpha value, b value, and lambda value
	public static sysConfigData readSysConfig(String filePath, boolean backTrack, boolean l2Norm, double alpha, double b, double lambda){
        FileReader readFile = null;
        BufferedReader inputScanner = null;
        try {
            readFile = new FileReader(filePath);
            inputScanner = new BufferedReader((readFile));

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        //sets up default values
        int iters = -1;
        int bad = -1;
        double step = -1;
        double lamb = -1;
        double alph = -1;
        double localB = -1;
        String line= null;
        try {
            //read every line one at a time.
            while((line=inputScanner.readLine())!=null){
                String [] tokens = line.trim().split("\\s+");
                //which token it is one at a time.
                if(tokens[0].equals("MAX_ITERS")){
                    iters = Integer.valueOf(tokens[1]);
                } else if(tokens[0].equals("MAX_BAD_COUNT")){
                    bad = Integer.valueOf(tokens[1]);
                } else if(tokens[0].equals("STEP_SIZE")){
                    step = Double.valueOf(tokens[1]);
                }else if(tokens[0].equals("LAMBDA")){
                    lamb = Double.valueOf(tokens[1]);
                }else if(tokens[0].equals("ALPHA")){
                    alph = Double.valueOf(tokens[1]);
                } else if(tokens[0].equals("B")){
                    localB = Double.valueOf(tokens[1]);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        //Override config file values with command line values
        if(backTrack){
            alph = alpha;
            localB = b;
        }
        if(l2Norm){
            lamb=lambda;
        }
        //creates and returns a sysConfigData class
		return new sysConfigData(iters, bad, step, lamb, alph, localB);
	}

}

//A helper class used in the program to be a wrapper around the general system config values.
class sysConfigData{
	public int MAX_ITERS = -1;
	public int MAX_BAD_COUNT = -1;
	public double STEP_SIZE = -1;
	public double LAMBDA = 0;
	public double ALPHA = -1;
	public double B = -1;
	public sysConfigData(int iters, int bad, double step, double lamb, double alph, double b ){
		this.MAX_ITERS = iters;
		this.MAX_BAD_COUNT = bad;
		this.STEP_SIZE = step;
		this.LAMBDA = lamb;
		this.ALPHA = alph;
		this.B = b;
	}
	
}

//A helper class used in the program to be a wrapper around the data information.
class trainDevConfig{
    public int N_TRAIN = 0;
    public int N_DEV = 0;
    public int D = 0;
    public int C = 0;
    public trainDevConfig(int train, int dev, int dim, int classes){
        this.N_TRAIN = train;
        this.N_DEV = dev;
        this.D = dim;
        this.C = classes;
    }
}

//a helper class so that I can return multiple Matrixes at once.
class matrixWrapper{
	public RealMatrix x;
	public RealMatrix y;
	public matrixWrapper(RealMatrix thatx, RealMatrix thaty){
		this.x = thatx;
		this.y = thaty;
	}
}

