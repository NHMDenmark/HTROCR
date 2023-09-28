from lxml import etree
from datetime import datetime
import os

def generate_xml(filename, polygon_coords, baseline_coords, region_coords, scale, transcription, outdir):
    root = etree.Element(
        "PcGts",
        nsmap={
            None: "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15",
            "xsi": "http://www.w3.org/2001/XMLSchema-instance"
        }
    )

    root.set(
        "{http://www.w3.org/2001/XMLSchema-instance}schemaLocation",
        "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd"
    )

    metadata = etree.SubElement(root, "Metadata")
    creator = etree.SubElement(metadata, "Creator")
    creator.text = "Natural History Museum of Denmark"
    created = etree.SubElement(metadata, "Created")
    current_time = datetime.now()
    created.text = current_time.strftime("%Y-%m-%dT%H:%M:%S.%f%z")

    page = etree.SubElement(root, "Page", imageFilename=filename)

    reading_order = etree.SubElement(page, "ReadingOrder")
    ordered_group = etree.SubElement(reading_order, "OrderedGroup", id="ro_1681373183487", caption="Regions reading order")
    for i in range(len(polygon_coords)):
        region_ref_indexed = etree.SubElement(ordered_group, "RegionRefIndexed", index=str(i), regionRef=f"region_{i}")

    for i in range(len(polygon_coords)):
        text_region = etree.SubElement(page, "TextRegion", type="paragraph", id=f"region_{i}", custom="readingOrder {index:"+str(i)+";} structure {type:paragraph;}")
        coords = etree.SubElement(text_region, "Coords", points=f"{int(region_coords[i][0])},{int(region_coords[i][2])} {int(region_coords[i][0])},{int(region_coords[i][3])} {int(region_coords[i][1])},{int(region_coords[i][3])} {int(region_coords[i][1])},{int(region_coords[i][2])}")

        text_line = etree.SubElement(text_region, "TextLine", id=f"line_{i}", custom="readingOrder {index:"+str(i)+";}")
        
        coords = etree.SubElement(text_line, "Coords", points=' '.join([f'{int(c[0])},{int(c[1])}' for c in polygon_coords[i]]))
        baseline = etree.SubElement(text_line, "Baseline", points=' '.join([f'{int(c[1]/scale)},{int(c[0]/scale)}' for c in baseline_coords[i]]))
        text_equiv = etree.SubElement(text_line, "TextEquiv")
        unicode_text = etree.SubElement(text_equiv, "Unicode")
        unicode_text.text = transcription[i]

    tree = etree.ElementTree(root)
    tree.write(os.path.join(outdir, f"{filename[:-4]}.xml"), pretty_print=True)
    
    
