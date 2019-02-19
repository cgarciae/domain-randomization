import os
from jinja2 import Environment, PackageLoader


class PascalVocWritter:
    def __init__(self, path, width, height, depth=3, database='Unknown', segmented=0):
        environment = Environment(keep_trailing_newline=True)
        self.annotation_template = environment.from_string(TEMPLATE)

        abspath = os.path.abspath(path)

        self.metadata = {
            'path': abspath,
            'filename': os.path.basename(abspath),
            'folder': os.path.basename(os.path.dirname(abspath)),
            'width': width,
            'height': height,
            'depth': depth,
            'database': database,
            'segmented': segmented,
            'objects': []
        }

    def add_object(self, name, xmin, ymin, xmax, ymax, pose='Unspecified', truncated=0, difficult=0):
        self.metadata['objects'].append({
            'name': name,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'pose': pose,
            'truncated': truncated,
            'difficult': difficult,
        })

    @property
    def xml_string(self):
        return self.annotation_template.render(**self.metadata)

    def save(self, annotation_path):
        with open(annotation_path, 'w') as file:
            content = self.xml_string()
            file.write(content)


TEMPLATE = """
<annotation>
    <folder>{{ folder }}</folder>
    <filename>{{ filename }}</filename>
    <path>{{ path }}</path>
    <source>
        <database>{{ database }}</database>
    </source>
    <size>
        <width>{{ width }}</width>
        <height>{{ height }}</height>
        <depth>{{ depth }}</depth>
    </size>
    <segmented>{{ segmented }}</segmented>
    {% for object in objects %}    
    <object>
        <name>{{ object.name }}</name>
        <pose>{{ object.pose }}</pose>
        <truncated>{{ object.truncated }}</truncated>
        <difficult>{{ object.difficult }}</difficult>
        <bndbox>
            <xmin>{{ object.xmin }}</xmin>
            <ymin>{{ object.ymin }}</ymin>
            <xmax>{{ object.xmax }}</xmax>
            <ymax>{{ object.ymax }}</ymax>
        </bndbox>
    </object>
    {% endfor %}
</annotation>
"""