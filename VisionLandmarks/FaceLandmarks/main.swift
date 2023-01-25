import Foundation
import Vision

guard (CommandLine.arguments.count == 2) else {
    print("Error: path to .jpg .jpeg or .png image expected")
    exit(-1)
}
let file = CommandLine.arguments[1]

var image: CGImage
if file.hasSuffix(".jpg") || file.hasSuffix(".jpeg") {
    let imgDataProvider = CGDataProvider(data: NSData(contentsOfFile: file)!)
    image = CGImage(jpegDataProviderSource: imgDataProvider!, decode: nil, shouldInterpolate: true, intent: CGColorRenderingIntent.defaultIntent)!
} else if file.hasSuffix(".png") {
    let imgDataProvider = CGDataProvider(data: NSData(contentsOfFile: file)!)
    image = CGImage(pngDataProviderSource: imgDataProvider!, decode: nil, shouldInterpolate: true, intent: CGColorRenderingIntent.defaultIntent)!
} else {
    print("Error: path to .jpg .jpeg or .png image expected")
    exit(-1)
}

let handler = VNImageRequestHandler(cgImage: image)
let request = VNDetectFaceLandmarksRequest() {req,err in
    if (err != nil) {
        print("error", err?.localizedDescription)
        return
    }
    var resultArray:[[String: Any]] = []
    for face in req.results! {
        let bbox = (face as! VNFaceObservation).boundingBox
        let landmarks = (face as! VNFaceObservation).landmarks!.allPoints!.normalizedPoints
        var points = []
        for l in landmarks {
            points.append([l.x, (1.0 - l.y)])
        }
        resultArray.append([
            "bbox": ["left": bbox.minX, "top": bbox.minY, "right": bbox.maxX, "bottom": bbox.maxY],
            "landmarks": points
        ])
    }
    let jsonData = try! JSONSerialization.data(withJSONObject: resultArray, options: [])
    let jsonString = NSString(data: jsonData, encoding: String.Encoding.utf8.rawValue)!
    print(jsonString)
}
try! handler.perform([request])
