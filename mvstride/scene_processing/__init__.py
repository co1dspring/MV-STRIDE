"""Scene annotation extraction & packaging.

Given generated (Infinigen) or captured/reconstructed (ScanNet++) scenes and
their rendered/captured images, extract the geometry / object / camera
annotations and package them into the unified ``scene_metadata.json`` format
consumed by the QA pipeline.

Planned layout (pending user guidance on file placement):

- ``infinigen/``   : extract ``scene_metadata.json`` from an Infinigen scene
                     inside Blender (moved from ``save_scene_annotation.py``).
- ``scannetpp/``   : convert raw ScanNet++ annotations into the unified schema
                     (moved from ``scannetpp2infinigen_new.py`` / ``repair_bbox.py``).
"""
