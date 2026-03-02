library(shiny)
library(plotly)

# ---- precompute data ----
x1 <- seq(-3, 3, length.out = 100)
x2 <- seq(-3, 3, length.out = 100)
grid <- expand.grid(x1 = x1, x2 = x2)

relu <- function(z) pmax(0, z)
z_linear <- 1*grid$x1 - 1*grid$x2
z1 <- relu( 1*grid$x1 - 1*grid$x2 )
z2 <- relu(-1*grid$x1 - 1*grid$x2 + 0.5)
z_relu <- 1*z1 - 1*z2

z_linear_mat <- matrix(z_linear, nrow = 100)
z_relu_mat   <- matrix(z_relu,   nrow = 100)
z1_mat       <- matrix(z1, 100)

# ---- UI ----
ui <- navbarPage(
  title = "Neural Network Explorer",
  
  tabPanel("Activation",
           sidebarLayout(
             
             sidebarPanel(
               width = 3,
               
               h4("🎥 Camera"),
               actionButton("btn_rotate", "▶ Rotate", class = "btn-success btn-block"),
               tags$br(),
               actionButton("btn_pause",  "⏸ Pause",  class = "btn-warning btn-block"),
               
               tags$hr(),
               
               h4("📐 Planes"),
               checkboxInput("show_linear", 
                             tags$span(style = "color:#4444ff; font-weight:bold;", "Linear Plane"), 
                             value = TRUE),
               checkboxInput("show_neuron", 
                             tags$span(style = "color:#22aa22; font-weight:bold;", "Hidden Neuron 1"), 
                             value = FALSE),
               checkboxInput("show_relu",   
                             tags$span(style = "color:#cc2222; font-weight:bold;", "ReLU Surface"),   
                             value = FALSE),
               
               tags$hr(),
               
               h4("ℹ️ Info"),
               conditionalPanel("input.show_linear",
                                div(class = "alert alert-info", style = "font-size:12px;",
                                    "The ", strong("linear plane"), " shows a standard linear transformation: f(x) = x1 - x2"
                                )
               ),
               conditionalPanel("input.show_neuron",
                                div(class = "alert alert-success", style = "font-size:12px;",
                                    strong("Hidden Neuron 1"), " applies ReLU to the linear combination: ReLU(x1 - x2)"
                                )
               ),
               conditionalPanel("input.show_relu",
                                div(class = "alert alert-danger", style = "font-size:12px;",
                                    "The ", strong("ReLU surface"), " is the network output combining both hidden neurons."
                                )
               )
             ),
             
             mainPanel(
               width = 9,
               plotlyOutput("plot3d", height = "650px")
             )
           )
  )
)

# ---- Server ----
server <- function(input, output, session) {
  
  output$plot3d <- renderPlotly({
    p <- plot_ly()
    
    if (input$show_linear) {
      p <- p %>% add_surface(
        x = x1, y = x2, z = z_linear_mat,
        opacity = 0.5,
        colorscale = list(c(0,1), c("blue","blue")),
        name = "Linear Plane",
        showscale = FALSE
      )
    }
    
    if (input$show_neuron) {
      p <- p %>% add_surface(
        x = x1, y = x2, z = z1_mat,
        opacity = 0.5,
        colorscale = list(c(0,1), c("green","green")),
        name = "Hidden Neuron 1",
        showscale = FALSE
      )
    }
    
    if (input$show_relu) {
      p <- p %>% add_surface(
        x = x1, y = x2, z = z_relu_mat,
        opacity = 0.6,
        colorscale = list(c(0,1), c("red","red")),
        name = "ReLU Surface",
        showscale = FALSE
      )
    }
    
    # fallback if nothing selected
    if (!input$show_linear && !input$show_neuron && !input$show_relu) {
      p <- p %>% add_annotations(
        text = "Select at least one plane from the sidebar",
        x = 0.5, y = 0.5, xref = "paper", yref = "paper",
        showarrow = FALSE, font = list(size = 16)
      )
    }
    
    p %>% layout(
      scene = list(
        xaxis = list(title = "x1"),
        yaxis = list(title = "x2"),
        zaxis = list(title = "f(x)"),
        camera = list(eye = list(x = 1.8, y = 0, z = 0.8))
      ),
      legend = list(x = 0.01, y = 0.99),
      margin = list(l = 0, r = 0, t = 30, b = 0)
    )
  })
}

# ---- JS for rotation (outside server, persists across re-renders) ----
ui$children[[1]]$children[[1]] <- tagAppendChild(
  ui$children[[1]]$children[[1]],
  tags$script(HTML("
    var rotateTimer = null;
    var angle = 0;

    function doRotate() {
      angle += 0.5;
      var rad = angle * Math.PI / 180;
      var plotEl = document.getElementById('plot3d');
      if (plotEl && plotEl._fullLayout) {
        Plotly.relayout(plotEl, {
          'scene.camera.eye': {
            x: 1.8 * Math.cos(rad),
            y: 1.8 * Math.sin(rad),
            z: 0.8
          }
        });
      }
    }

    $(document).on('click', '#btn_rotate', function() {
      if (!rotateTimer) {
        rotateTimer = setInterval(doRotate, 30);
      }
    });

    $(document).on('click', '#btn_pause', function() {
      clearInterval(rotateTimer);
      rotateTimer = null;
    });
  "))
)

shinyApp(ui, server)