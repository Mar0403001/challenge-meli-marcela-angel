"""Limpia el HTML crudo que aparece embebido dentro de archivos Markdown.

Varios proyectos (catalog-portfolio-api, traffic-gate-api, order-workflow-api) tienen
bloques de HTML completo mezclados con el Markdown: tarjetas de navegación
(`<div class="content-card">...`), contenido colapsable (`<details><summary>...`), o
incluso un README que arranca directo en HTML puro, sin ningún título en Markdown.
Para el corpus de texto ese HTML no aporta nada como formato -- lo único que importa
es el texto que leería una persona, no las etiquetas. Este módulo lo reduce a texto
plano.
"""

from __future__ import annotations

# BeautifulSoup (el paquete se instala como "beautifulsoup4", pero se importa como
# "bs4") es una libreria para leer HTML "sucio" o incompleto y poder navegarlo o
# extraer texto de el, sin tener que escribir un parser de HTML a mano.
from bs4 import BeautifulSoup


def html_block_to_text(html: str) -> str:
    """Convierte un bloque de HTML crudo (ej. `<div class="content-card">...`) en
    texto plano legible, dejando un espacio entre una etiqueta y otra (para que no
    queden dos palabras pegadas, tipo "TituloTexto").

    Se usa el parser 'html.parser' que ya trae Python (no una libreria externa como
    lxml) porque el HTML de docs_raw es simple -- divs, details, imagenes, links, sin
    tablas HTML complejas ni casos raros -- y el parser mas liviano alcanza sin
    perder nada, sin sumar una dependencia extra que ademas requeriria compilar codigo C.
    """
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(" ", strip=True)
