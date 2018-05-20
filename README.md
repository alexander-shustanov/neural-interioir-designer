# NeuralInteriorDesigner
<p>NeuralInteriorDesigner - приложение для дизайнеров, которое переносит работу с изображениями на новый уровень.</p>

<p><img alt="" src="https://github.com/alexander-shustanov/neural-interioir-designer/blob/master/presentation/LOGO.jpg?raw=true" /></p>

<p>Идея приложения - с помощью простых инструментов менять облик комнаты, таким образом как: изменение обоев, потолка, пола;&nbsp;замена одних элементов интерьера на другие, удаление&nbsp;элементов интерьера, добавление элементов интерьера.</p>

<p>Отдельные части приложения в том или ином виде готовы, но в итоговое приложение объединены следующие компоненты: изменение обоъев, пола, потолка; &quot;умная волшебная палочка&quot;. Остальные части продемонстрированы в jupyter ноутбуках.</p>

<p>Ноу-хау: &quot;Умная волшебная палочка&quot;. Новый метод выделения области изображения, который работает не с пикселями, а с объектами.&nbsp;</p>

<p>Для создания генеративных моделей использовался датасет <a href="http://rgbd.cs.princeton.edu/">SUNRGBD</a>.&nbsp;</p>

<p>Примеры:&nbsp;</p>

<p>1. Удаление элементов интерьера. В&nbsp;приложение не добавлено, работает из jupyter-notebook.</p>

<table align="center" border="0" cellpadding="0" cellspacing="0">
	<tbody>
		<tr>
			<td><img alt="original" src="https://github.com/alexander-shustanov/neural-interioir-designer/blob/master/presentation/removing_original.jpg?raw=true" style="height:300px; width:225px" width="300" height="300" /></td>
			<td><img alt="altered" src="https://github.com/alexander-shustanov/neural-interioir-designer/blob/master/presentation/altered.jpg?raw=true" style="height:327px; width:331px" /></td>
			<td><img alt="result" src="https://github.com/alexander-shustanov/neural-interioir-designer/blob/master/presentation/without_chair.jpg?raw=true" style="height:300px; width:295px" /></td>
		</tr>
	</tbody>
</table>

<p>&nbsp;</p>

<p>2. Умная волшебная палочка. Добавлено в приложение.</p>

<p><img alt="magic" src="https://github.com/alexander-shustanov/neural-interioir-designer/blob/master/presentation/application/Screenshot%20from%202018-05-19%2015-33-51.png?raw=true" style="height:314px; width:323px" /></p>

<p>3. Добавление элементов интерьера. Работает плохо, только из jupyter-notebook. По всей видимости имеет место переобучение, требуются дальнейшие исследования.</p>

<table align="center" border="0" cellpadding="0" cellspacing="0">
	<tbody>
		<tr>
			<td><img alt="input" src="https://github.com/alexander-shustanov/neural-interioir-designer/blob/master/presentation/generative/1_input.png?raw=true" style="height:256px; width:256px" /></td>
			<td><img alt="result" src="https://github.com/alexander-shustanov/neural-interioir-designer/blob/master/presentation/generative/1_output.png?raw=true" style="height:256px; width:256px" /></td>
			<td><img alt="target" src="https://github.com/alexander-shustanov/neural-interioir-designer/blob/master/presentation/generative/1_target.png?raw=true" style="height:256px; width:256px" /></td>
		</tr>
	</tbody>
</table>

<p>4. Замена обоев, полов. Основано на немного видоизмененном style-tranfer. Добавлено в приложение.</p>

<p><img alt="" src="https://github.com/alexander-shustanov/neural-interioir-designer/blob/master/presentation/application/Screenshot%20from%202018-05-19%2015-34-16.png?raw=true" /></p>

<p><img alt="" src="https://github.com/alexander-shustanov/neural-interioir-designer/blob/master/presentation/application/new_wallpaper.png?raw=true" /></p>

Также смотрите [презентацию](https://docs.google.com/presentation/d/1qarSe9f87gavkwd_HIe4h0qx93R9wYjG-r5xJf9HhbI/edit?usp=sharing) и [видео](https://vimeo.com/270951743).

# Инструкции

Для запуска приложения выполните. 

```bash
python3 app.py
```

Для запуска модели в [jupyter-notebook](https://github.com/alexander-shustanov/neural-interioir-designer/blob/master/pix2pix/generate.ipynb) скачайте [модели](https://drive.google.com/drive/folders/1-s_thwO3ZPERqZyWJKKJtvL0et9lbe3L) и распакуйте в папку pix2pix.


# Источники

[pix2pix](https://github.com/affinelayer/pix2pix-tensorflow)

[SUNRGBD датасет](http://rgbd.cs.princeton.edu/)

