var canvas = document.querySelector('canvas');
var context = canvas.getContext('2d')
var otrocanvas = document.querySelector('#otrocanvas')
console.log(otrocanvas)
var upload = document.querySelector('#upload')


upload.addEventListener('change', (e)=>{
    const myFile = upload.files[0];
    const img = new Image();
    img.src = URL.createObjectURL(myFile);
    img.onload = function(){
        context.drawImage(img,0,0,canvas.width,canvas.height)
        otrocanvas.getContext('2d').drawImage(img, 0, 0, otrocanvas.width, otrocanvas.height)   
        procesarImage()    
}
})

var modelo = null;

(async() => {
  console.log("Cargando modelo...");
  modelo = await tf.loadLayersModel("modelo_v2/model.json");
  console.log("Modelo cargado");
})();

function procesarImage(){
        context.drawImage(otrocanvas, 0, 0, 0, 0);
}


function predecir() {

        if (modelo != null) {
          resample_single(canvas, 200, 200, otrocanvas);
  
          //Hacer la predicción
          var ctx2 = otrocanvas.getContext("2d");
          var imgData = ctx2.getImageData(0,0, 200, 200);
  
          var arr = [];
          var arr200 = [];
  
          for (var p=0; p < imgData.data.length; p+= 4) {
            var rojo = imgData.data[p] / 255;
            var verde = imgData.data[p+1] / 255;
            var azul = imgData.data[p+2] / 255;
  
            var gris = (rojo+verde+azul)/3;
  
            arr200.push([gris]);
            if (arr200.length == 200) {
              arr.push(arr200);
              arr200 = [];
            }
          }
  
          arr = [arr];
  
          var tensor = tf.tensor4d(arr);
          var resultado = modelo.predict(tensor).dataSync();
          var max = Math.max(...resultado);
          var postion = resultado.indexOf(max)
        console.log("entro")
        if(postion == 1){
                     respuesta = 'Carlos'
                }
                else if(postion == 2){
                                respuesta = 'German'
                        }
                        else if(postion == 3)
                        {
                                respuesta = 'Omar'
                        }
          var respuesta;
          console.log(resultado)
          console.log("prediccion: " + resultado)
        }
        document.getElementById("request").innerHTML = respuesta
        document.getElementById("request2").innerHTML = max.toFixed(2)
      }

 /**
       * Hermite resize - fast image resize/resample using Hermite filter. 1 cpu version!
       * 
       * @param {HtmlElement} canvas
       * @param {int} width
       * @param {int} height
       * @param {boolean} resize_canvas if true, canvas will be resized. Optional.
       * Cambiado por RT, resize canvas ahora es donde se pone el chiqitillllllo
       */
      function resample_single(canvas, width, height, resize_canvas) {
          var width_source = canvas.width;
          var height_source = canvas.height;
          width = Math.round(width);
          height = Math.round(height);

          var ratio_w = width_source / width;
          var ratio_h = height_source / height;
          var ratio_w_half = Math.ceil(ratio_w / 2);
          var ratio_h_half = Math.ceil(ratio_h / 2);

          var ctx = canvas.getContext("2d");
          var ctx2 = resize_canvas.getContext("2d");
          var img = ctx.getImageData(0, 0, width_source, height_source);
          var img2 = ctx2.createImageData(width, height);
          var data = img.data;
          var data2 = img2.data;

          for (var j = 0; j < height; j++) {
              for (var i = 0; i < width; i++) {
                  var x2 = (i + j * width) * 4;
                  var weight = 0;
                  var weights = 0;
                  var weights_alpha = 0;
                  var gx_r = 0;
                  var gx_g = 0;
                  var gx_b = 0;
                  var gx_a = 0;
                  var center_y = (j + 0.5) * ratio_h;
                  var yy_start = Math.floor(j * ratio_h);
                  var yy_stop = Math.ceil((j + 1) * ratio_h);
                  for (var yy = yy_start; yy < yy_stop; yy++) {
                      var dy = Math.abs(center_y - (yy + 0.5)) / ratio_h_half;
                      var center_x = (i + 0.5) * ratio_w;
                      var w0 = dy * dy; //pre-calc part of w
                      var xx_start = Math.floor(i * ratio_w);
                      var xx_stop = Math.ceil((i + 1) * ratio_w);
                      for (var xx = xx_start; xx < xx_stop; xx++) {
                          var dx = Math.abs(center_x - (xx + 0.5)) / ratio_w_half;
                          var w = Math.sqrt(w0 + dx * dx);
                          if (w >= 1) {
                              //pixel too far
                              continue;
                          }
                          //hermite filter
                          weight = 2 * w * w * w - 3 * w * w + 1;
                          var pos_x = 4 * (xx + yy * width_source);
                          //alpha
                          gx_a += weight * data[pos_x + 3];
                          weights_alpha += weight;
                          //colors
                          if (data[pos_x + 3] < 255)
                              weight = weight * data[pos_x + 3] / 250;
                          gx_r += weight * data[pos_x];
                          gx_g += weight * data[pos_x + 1];
                          gx_b += weight * data[pos_x + 2];
                          weights += weight;
                      }
                  }
                  data2[x2] = gx_r / weights;
                  data2[x2 + 1] = gx_g / weights;
                  data2[x2 + 2] = gx_b / weights;
                  data2[x2 + 3] = gx_a / weights_alpha;
              }
          }


          ctx2.putImageData(img2, 0, 0);
      }