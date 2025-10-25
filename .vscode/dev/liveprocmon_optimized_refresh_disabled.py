
def generate_index_html():

    with open('index.html', 'w') as index_html:
        index_html.write("<!DOCTYPE html><body><header>")
        index_html.write('<script>')
        index_html.write('var counter = 0;')
        index_html.write('var plotimg = document.getElementById("plotimg");')
        index_html.write('function refreshPlotImg()')
        index_html.write('{')
        index_html.write('    plotimg.src = plotimg.src + "?" + (counter++);')
        index_html.write('    setTimeout(refreshPlotImg, {})'.format(UPDATE_FREQUECY_MS))
        index_html.write('}')
        index_html.write('refreshPlotImg()')
        index_html.write('</script></header>')
        index_html.write('<img id="plotimg" src="{}">'.format(PLOT_FILE_NAME))
        index_html.write("</body></html>")
