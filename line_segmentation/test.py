from lxml import etree
from datetime import datetime

def generate_xml(filename, img_shape, polygon_coords, baseline_coords, transcription):

    # Create the root element and set the XML namespace
    root = etree.Element(
        "PcGts",
        nsmap={
            None: "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15",
            "xsi": "http://www.w3.org/2001/XMLSchema-instance"
        }
    )

    # Add the schemaLocation attribute
    root.set(
        "{http://www.w3.org/2001/XMLSchema-instance}schemaLocation",
        "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd"
    )

    # Create the Metadata element
    metadata = etree.SubElement(root, "Metadata")
    creator = etree.SubElement(metadata, "Creator")
    creator.text = "Natural History Museum of Denmark"
    created = etree.SubElement(metadata, "Created")
    current_time = datetime.now()
    created.text = current_time.strftime("%Y-%m-%dT%H:%M:%S.%f%z")

    # Create the Page element
    page = etree.SubElement(root, "Page", imageFilename="sp61019672878142371646.att.jpg", imageWidth="6818", imageHeight="9287")

    # Create the ReadingOrder element
    reading_order = etree.SubElement(page, "ReadingOrder")
    ordered_group = etree.SubElement(reading_order, "OrderedGroup", id="ro_1681373183487", caption="Regions reading order")
    region_ref_indexed = etree.SubElement(ordered_group, "RegionRefIndexed", index="0", regionRef="region_1681373077334_0")

    # Create the TextRegion element
    text_region = etree.SubElement(page, "TextRegion", type="paragraph", id="region_1681373077334_0", custom="readingOrder {index:0;} structure {type:paragraph;}")
    coords = etree.SubElement(text_region, "Coords", points="205,9091 205,9150 600,9150 600,9091")

    # Create the TextLine element
    text_line = etree.SubElement(text_region, "TextLine", id="line_1681373077362_33", custom="readingOrder {index:0;}")
    coords = etree.SubElement(text_line, "Coords", points="206,9091 600,9100 599,9150 205,9141")
    baseline = etree.SubElement(text_line, "Baseline", points="205,9136 599,9145")
    text_equiv = etree.SubElement(text_line, "TextEquiv")
    unicode_text = etree.SubElement(text_equiv, "Unicode")
    unicode_text.text = ""

    # Add TextEquiv to TextRegion as well
    text_equiv = etree.SubElement(text_region, "TextEquiv")
    unicode_text = etree.SubElement(text_equiv, "Unicode")
    unicode_text.text = ""

    # Create the XML tree and write to a file
    tree = etree.ElementTree(root)
    tree.write("output.xml", pretty_print=True)

strinas ="./line_segmentation/demo/orig.jpg"
print(strinas.split('/')[-1])
import numpy as np
print(np.zeros((100, 256)).T.shape)