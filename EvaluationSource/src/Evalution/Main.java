/*
 * This code refers http://imagej.net/Segmentation_evaluation_after_border_thinning_-_Script
 */

package Evalution;
import ij.ImagePlus;
import trainableSegmentation.metrics.*;

public class Main {

    public static void main(String[] args) {
        if (args.length !=2) {
            System.out.append("Place give 2 arguments, first is true images, second is target images" + System.lineSeparator());
            return;
        }

        Boolean binaryProposal = true;
        double maxThres = binaryProposal ? 0.0 : 1.0;
        ImagePlus trueImage = new ImagePlus(args[0]);
        ImagePlus targetImage = new ImagePlus(args[1]);

        System.out.append("Calculating PixelError" + System.lineSeparator());
        PixelError pixelError = new PixelError( trueImage, targetImage );
        pixelError.getPixelErrorMaximalFScore( 0.0, maxThres, 0.1);

        System.out.append("Calculating RandError" + System.lineSeparator());
        RandError randError = new RandError( trueImage, targetImage );
        randError.getMaximalVRandAfterThinning( 0.0, maxThres, 0.1, true );

        System.out.append("Calculating VariationOfInformation" + System.lineSeparator());
        VariationOfInformation variationOfInformation = new VariationOfInformation( trueImage, targetImage );
        variationOfInformation.getMaximalVInfoAfterThinning( 0.0, maxThres, 0.1 );
    }
}
