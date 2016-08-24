//David Shagam
//Program 1
//Machine learning
//CSCI 497E

//General imports
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Scanner;
//Apache imports
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;


public class Prog1{
	
	public static void main(String[] args){
		parseArgs(args);
		System.out.println("Program Completed");
		
	}
	
	
	//ALL expect to end in N D K. 
	//Train 2 txt files, output file, and multiple choices
	//predict is simple
	//eval is simple.
	private static void parseArgs(String[] args){
		//Not enough args
		if(args.length < 7){
			throw new IllegalArgumentException("Not enough arguments passed expected a minimum of 7 recieved " + args.length);
		} else {
			
			//File string paths. There are ALWAYS 3
			String filePath1 = args[1];
			String filePath2 = args[2];
			String filePath3 = args[3];
			
			//Used for train with gradient descent
			double stepSize = 0.1;
			double stopThreshold = 0;
			
			//Always present defines the information about the datasets.
			int polynomialOrder = Integer.parseInt(args[args.length-1]);
			int dimensionality = Integer.parseInt(args[args.length-2]);
			int numLines = Integer.parseInt(args[args.length-3]);
			//Error if dimensions and poly are greater than 1. we don't support this yet.
			if(polynomialOrder > 1 && dimensionality > 1){
				throw new IllegalArgumentException("If polynomial order is greater than 1 then the dimensionality of the data must be 1.");
			}
			//Init x data matrix
			RealMatrix xMatrix = null;
			//Polynomial logic for loading x.
			if(polynomialOrder == 1){
				xMatrix = loadData(filePath1, numLines, dimensionality+1, true, 1);
			} else if(polynomialOrder > 1 ) {
				xMatrix = loadData(filePath1, numLines, polynomialOrder+1, true, polynomialOrder);
				dimensionality = polynomialOrder;
			}
			
			//which one are we dealing with.
			if( args[0].equals("-train")){
				//load y data
				RealMatrix yMatrix = loadData(filePath2, numLines, 1);
				//closed form
				if(args[4].equals("a")){
					//check for enough args
					if(args.length < 8){
						throw new IllegalArgumentException("Not enough arguments present when using flag a expects 8 total");
					}
					//exceute closed form.
					closedForm(xMatrix, yMatrix, polynomialOrder, filePath3);
					
				//Gradient descent
				} else if(args[4].equals("g")){
					//check for enough args
					if(args.length < 10){
						throw new IllegalArgumentException("Not enough arguments present when using flag g need 10 arguments total");
					}
					//grabs step size and stop threshold.
					stepSize = Double.parseDouble(args[5]);
					stopThreshold = Double.parseDouble(args[6]);
					//execute closed form.
					gradientDescent(xMatrix, yMatrix, stepSize, stopThreshold, numLines, polynomialOrder,dimensionality, filePath3 );
					
				} else {
					throw new IllegalArgumentException("Expected either a or g but got " + args[4]);
				}
				
			}else if( args[0].equals("-pred")){
				//deal with prediction
				//Load model
				RealMatrix model = loadModel(filePath2, dimensionality+1);
				//run prediction
				predictValues(xMatrix, model, filePath3);
			}else if( args[0].equals("-eval")){
				//load data
				RealMatrix yMatrix = loadData(filePath2, numLines, 1);
				//load model
				RealMatrix model = loadModel(filePath3, dimensionality+1);
				//evaluates model
				evaluateModel(xMatrix, yMatrix, model, numLines, true);
			} else {
				throw new IllegalArgumentException("Expected -train, -pred, or -eval, got " + args[0]);
			}
			
		}

		
	}
	
	
	//Computes the closed form using matrix algrebra for linear regression.
	//Takes 2 matrixes the data points x and the associated value y
	//Takes int with the polynomial order
	//Takes a string with a file path to write to.
	private static void closedForm(RealMatrix x, RealMatrix y, int poly, String Output){
		//Equation ((A^T A)^-1) A^T B
		// C = A^T A
		// D = A^T B
		// E = C^-1
		// F = D E
		RealMatrix C = (x.transpose()).multiply(x);
		RealMatrix D = (x.transpose()).multiply(y);
		RealMatrix E = null;
		try{
			E = MatrixUtils.inverse(C);
		} catch(Exception e ){
			System.out.println("Matrix cannot be inverted please try gradient descent");
			throw e;	
		}
		RealMatrix result = E.multiply(D);
		//Print the values we got
		outputModel(result, Output);
	}
	
	//Calculates the gradient descent for linear regression
	//Takes 2 matrixes the X points and the assocaited Y values.
	//Takes 2 doubles the step size and the stop threshold.
	//takes 3 Ints the number of lines in the file, the polynomial dimension, the matrix dimension
	//Takes a string with where to write the model.
	private static void gradientDescent(RealMatrix x, RealMatrix y, double step, double stop, int lines, int poly, int Dim, String Output){
		//Derivative is = 2/lines * sum of each line (estimate - actual) * x
		
		//setup default state
		boolean converged = false;
		double[] beta = new double[Dim+1];
		//init beta to 0
		for(int i=0; i < Dim+1; i++){
			beta[i] = 0;
		}
		//creates the matrix
		RealMatrix betaReal = MatrixUtils.createColumnRealMatrix(beta);
		//counts iterations
		int its = 1;
		//Old error set to max so we dont' instantly quit
		double oldError = Double.MAX_VALUE;
		
		while (!converged){
			
			//Calculate gradient
			RealMatrix grad = (x.transpose()).multiply(y);
			grad = grad.scalarMultiply(-2.0/lines);
			RealMatrix steps2 = (x.transpose()).multiply(x);
			steps2 = steps2.multiply(betaReal);
			steps2 = steps2.scalarMultiply(2.0/lines);
			grad = grad.add(steps2);
			
			//Updates Beta
			betaReal = betaReal.subtract( grad.scalarMultiply(step));
			
			//Evaulate how we did
			double newError = evaluateModel(x, y, betaReal, lines, false);
			
			//Check for convergence
			converged = checkConvergence(oldError, newError, stop);
			oldError = newError;
			its++;
		}
		System.out.println("Gradient Descent ran " + its +" times");
		//Printout the model
		outputModel(betaReal, Output);
	}
	
	//A funciton to check if convergence is enough to quit.
	//Takes the values to compare and the stopping threshold.
	//returns true if converged and false otherwise.
	private static boolean checkConvergence(double oldVal, double newVal, double stop){	
		double relative = (oldVal - newVal) / oldVal;
		if(relative <= stop){
			return true;
		}
		
		return false;
	}
	
	//Helper function to print a model out to a file Used for closed form and gradient descent.
	//Takes a matrix and a string
	//Outputs a file at the path indicated by the string consisting of scientific notation doubles.
	private static void outputModel(RealMatrix beta, String fileName){
		//create file
		File outputFile = new File(fileName);
		try {
			outputFile.createNewFile();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			System.out.println("Unable to create file");
			e.printStackTrace();
			System.exit(1);
		}
		//create writer
		BufferedWriter Writer = null;
	
		try {
			//buffered writer
			Writer = new BufferedWriter(new FileWriter(outputFile));
			//Get the data to print out.
			double[] dataToWrite = beta.getColumn(0);
			String toWrite = "";
			for(int i = 0; i < dataToWrite.length; i++){
				//convert double to scientific notation.
				toWrite = toWrite + toScientific(dataToWrite[i]) + " ";
			}
			//flush out to the file.
			Writer.write(toWrite.trim());
			Writer.flush();
			Writer.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			System.out.println("Unable to create buffered Writer or File Writer");
			e.printStackTrace();
			System.exit(1);
		}
	}
	
	//Loads the model.
	//Takes the path to the model to load and the dimension of the file.
	//returns a matrix representing the model beta coefficeints.
	private static RealMatrix loadModel(String path, int Dim ){
		double[] loadModel = new double[Dim];
		//open the file
		File readFile = new File(path);
		Scanner inputScanner = null;
		try{
			//get the data from the file
			inputScanner = new Scanner(readFile);
		} catch (FileNotFoundException e){
			e.printStackTrace();
		}
		for(int i = 0; i<Dim; i++){
			//read every double
			loadModel[i] = inputScanner.nextDouble();
		}
		
		return MatrixUtils.createColumnRealMatrix(loadModel);
	}
	
	//Wrapper for loading y matrix values;
	//returns a matrix
	private static RealMatrix loadData(String file, int lines, int dim){
		//file path, num lines, dimension of file, not x values, polynomial dim
		return loadData(file, lines, dim, false, 1);
	}
	
	//Loads data from files handles both x and y. has special logic for if were loading xVals and special logic for polynomials
	//Takes file path, num lines, num dimension, if were grabbing x or y vals, and the poly order
	//returns a matrix
	private static RealMatrix loadData(String file, int Lines, int Dim, boolean xVals, int polyOrder){
		//init double array with dimensions
		double[][] loadMatrix = new double[Lines][Dim];
		
		//Load a file reader and buffer for efficient file reading
		FileReader readFile = null;
		BufferedReader inputScanner =null;
		//Scanner lineScanner = null;
		try {
			//open the file and the buffer
			readFile = new FileReader(file);
			inputScanner = new BufferedReader(readFile);
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		//For every line do things
		for(int n=0; n<Lines; n++){
			String lineOfFile = null;
			
			try {
				//grab the line
				lineOfFile = inputScanner.readLine();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			//tokenize the line
			String[] tokens = lineOfFile.split(" ");
			
			//init d to default
			int d = 0;
			int tokenNum = 0;
			//if were loading the x values we need a column of 1's
			if(xVals){
				loadMatrix[n][0]=1;
				d=1;
			}
			if(polyOrder == 1){
				for(;d< Dim; d++ ){
					//load the double value of the strings into the array
					loadMatrix[n][d] = Double.valueOf(tokens[tokenNum]);
					tokenNum++;
				}
			} else {
				//Special logic for higher order polynomials.
				double loadVal = Double.valueOf(tokens[0]);
				for(;d< Dim; d++){
					loadMatrix[n][d] = Math.pow(loadVal, d);
				}
			}
		}
		//create and return matrix
		return MatrixUtils.createRealMatrix(loadMatrix);
	}
	
	
	//Predicts the values of a model given input. and outputs a file consiting of predictions to Path
	private static void predictValues(RealMatrix x, RealMatrix model, String path){
		//Simple matrix algebra to make predictions
		RealMatrix result = x.multiply(model);
		outputPrediction(result, path);
	}
	
	//Writes out a file of our predicted results.
	//Takes a matrix and the path to write to
	//Result file written at path.
	private static void outputPrediction(RealMatrix x, String path){
		File outputFile = new File(path);
		try {
			outputFile.createNewFile();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		//buffered writer set up
		BufferedWriter outputBuff = null;
		
		try {
			outputBuff = new BufferedWriter(new FileWriter(outputFile));
			//get the double values
			double[] outputVals = x.getColumn(0);
			for(int i =0; i< outputVals.length; i++){
				//write the values in scientific notation.
				outputBuff.write(toScientific(outputVals[i]).trim());
				outputBuff.newLine();
				
			}
			outputBuff.flush();
			outputBuff.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	//Evaulates our model to see if it's a good fit.
	//This outputs to standard out the MSE.
	//Takes dev data points, takes "true" values, our model, the number of lines, and if we should output to standard out.
	//Returns a double for use in gradient descent.
	private static double evaluateModel(RealMatrix x, RealMatrix y, RealMatrix model, int numLines, boolean doPrint){
		//get hte prediction.
		RealMatrix predictedX = x.multiply(model);
		predictedX = predictedX.subtract(y);
		//grab the data of our prediction
		double[][] sumMatrix = predictedX.getData();
		//get the sum of the squares
		double sum = 0;
		for(int i= 0; i < numLines; i++){
			double val = sumMatrix[i][0];
			val = val * val;
			sum += val;
		}
		//average the value
		sum = sum / numLines;
		if(doPrint){
			System.out.println("The MSE is: " + toScientific(sum));
		}
		return sum;
	}

	//Helper function to convert doubles to string scientific notation representations of themselves
	//Takes a double that we are converting to a string 
	//returns a string
	private static String toScientific(double val){
		//setup a number formatter
		NumberFormat scientific = new DecimalFormat("0.###E0");
		//returns scientific notation equivalent.
		return scientific.format(val);
	}
}
