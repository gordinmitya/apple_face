Use Apple's Vision framework (VNDetectFaceLandmarksRequest) to annotate faces on 300W, celeba_hq, ffhq, WFLW datasets.
And then to train lightweight landmarks model to have the same output format.

So one can reuse code that expects Apple's unique 78-point landmark format in Android applications. (or anywhere else)