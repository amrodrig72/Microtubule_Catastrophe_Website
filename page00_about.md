---
layout: page
title: About
description: Information about the project, website, and links to the paper and SI
img: microtubule.png # Add image post (optional)
caption: "Kinesin walking across a microtubule, from The Inner Life of a Cell by Cellular Visions and Harvard"
permalink: index.html
sidebar: true
---

---


# {{site.data.about.title}}
{{site.data.about.authors}}

{% for entry in site.data.about %}

{% if entry[0] != 'title' %}
{% if entry[0] != 'authors' %}
## {{entry[0]}}
{{entry[1]}}
{% endif %}
{% endif %}
{% endfor %}