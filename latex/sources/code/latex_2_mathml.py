import latex2mathml.converter

latex_input = str(input())
mathml_output = latex2mathml.converter.convert(latex_input)

print(mathml_output)