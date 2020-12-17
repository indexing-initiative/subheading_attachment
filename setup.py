import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="subheading_attachment_model_trainer",
    version="1.0.0",
    author="Alastair Rae",
    author_email="alastair.rae@nih.gov",
    description="Package for training subheading attachment models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["subheading_attachment_model_trainer", "subheading_attachment_model_trainer.cnn",]
)